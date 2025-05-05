import matplotlib.pyplot as plt
import ipyleaflet as L
from ipyleaflet import AwesomeIcon, Marker, Map
import ipywidgets as widgets
from branca.colormap import linear
import xarray as xr
import numpy as np
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  
from shiny.types import SafeException
from shiny.types import SilentException
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pour le calcul des proba
import sys,os
import pickle as pk
import itertools as itt
import datetime as dt

## Scientific libraries
##=====================
import numpy as np
import scipy.stats as sc
import SDFC as sd
import pandas as pd
import xarray as xr
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm
from cmdstanpy import cmdstan_path, set_cmdstan_path

## Plot libraries
##===============
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mplpatch
import matplotlib.gridspec as mplgrid
import matplotlib.backends.backend_pdf as mpdf

# SDFC a tendance a afficher plein de warnings pendant les fits : on les désactive
import warnings
warnings.simplefilter("ignore")

################################################
## Fonctions calcul ##
################################################

def plink( x , e = 2 / 3 ):
	y = np.arctan( np.log(x) ) / (np.pi / 2)
	return np.power( 1 + y , e )

def PRlink( x , e = 3 ):
	y = np.arctan( np.log(x ) ) / (np.pi / 2)
	return np.sign(y) * np.power( np.abs(y) , e )

def correct_miss( X , lo =  100 , up = 350 ):##{{{
	#	return X
	mod = str(X.columns[0])
	bad = np.logical_or( X < lo , X > up )
	bad = np.logical_or( bad , np.isnan(X) )
	bad = np.logical_or( bad , np.logical_not(np.isfinite(X)) )
	if np.any(bad):
		idx,_ = np.where(bad)
		idx_co = np.copy(idx)
		for i in range(idx.size):
			j = 0
			while idx[i] + j in idx:
				j += 1
			idx_co[i] += j
		X.iloc[idx] = X.iloc[idx_co].values
	return X
##}}}

def load_obs( path, ilat, ilon ):##{{{
	
	## Load Xo
	dXo_full = pd.read_csv(os.path.join( path , 'Xo/HadCRUT5_GSAT.csv'))
	year_Xo = dXo_full.loc[:,"Time"].values
	dXo_sub = dXo_full.drop(["Fraction of area represented", "Coverage uncertainty (1 sigma)"],axis=1)
	dXo = dXo_sub.median(axis=1)
	Xo  = pd.DataFrame( dXo.values.squeeze() , index = year_Xo )
	
	dYo_full   = xr.open_dataset( os.path.join( path , "Yo/tx3d/tx3d_era5_1940-2022_g025.nc" ) )
	dYo = dYo_full.sel(lat=ilat,lon=ilon,method="nearest")
	year_Yo = dYo.time["time.year"].values
	Yo    = pd.DataFrame( dYo.tasmax.values.ravel() , index = year_Yo )
    
	lat = dYo.lat.values
	lon = dYo.lon.values
 
	return Xo,Yo,lat,lon

##}}}

class NumpyLog: ##{{{
	def __init__(self):
		self._msg = []

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return "".join(self._msg)

	def write( self , msg ):
		self._msg.append(msg)
##}}}

def split_into_valid_time( nan_values , ci ):##{{{
	time_valid    = []
	time_notvalid = []
	t = int(nan_values.time[0])
	on_valid      = bool(nan_values.loc[t] < ci)
	curr = [t]
	
	for t in nan_values.time[1:]:
		curr.append(int(t))
		is_valid = bool(nan_values.loc[t] < ci)
		if not is_valid == on_valid:
			if on_valid: time_valid.append(curr)
			else: time_notvalid.append(curr)
			on_valid = not on_valid
			curr = [int(t)]
	if on_valid: time_valid.append(curr)
	else: time_notvalid.append(curr)
	
	return time_valid,time_notvalid


################################################
## Fonctions shiny ##
################################################


def split_contours(segs, kinds=None):
	"""takes a list of polygons and vertex kinds and separates disconnected vertices into separate lists.
	The input arrays can be derived from the allsegs and allkinds atributes of the result of a matplotlib
	contour or contourf call. They correspond to the contours of one contour level.
	
	Example:
	cs = plt.contourf(x, y, z)
	allsegs = cs.allsegs
	allkinds = cs.allkinds
	for i, segs in enumerate(allsegs):
		kinds = None if allkinds is None else allkinds[i]
		new_segs = split_contours(segs, kinds)
		# do something with new_segs
		
	More information:
	https://matplotlib.org/3.3.3/_modules/matplotlib/contour.html#ClabelText
	https://matplotlib.org/3.1.0/api/path_api.html#matplotlib.path.Path
	"""
	if kinds is None:
		return segs    # nothing to be done
	# search for kind=79 as this marks the end of one polygon segment
	# Notes: 
	# 1. we ignore the different polygon styles of matplotlib Path here and only
	# look for polygon segments.
	# 2. the Path documentation recommends to use iter_segments instead of direct
	# access to vertices and node types. However, since the ipyleaflet Polygon expects
	# a complete polygon and not individual segments, this cannot be used here
	# (it may be helpful to clean polygons before passing them into ipyleaflet's Polygon,
	# but so far I don't see a necessity to do so)
	new_segs = []
	for i, seg in enumerate(segs):
		segkinds = kinds[i]
		boundaries = [0] + list(np.nonzero(segkinds == 79)[0])
		for b in range(len(boundaries)-1):
			new_segs.append(seg[boundaries[b]+(1 if b>0 else 0):boundaries[b+1]])
	return new_segs

set_seed: np.random.seed(1)

year = list(range(1850, 2101))

zoom = 4
lat1 = 43 #43.6
lon1 = 19 #1.43
center = [lat1, lon1]

variables = ["maximum temperature"]
durations = ["3"]

## Path
##=====
basepath = os.getcwd() 
pathInp  = os.path.join( basepath , "data/"  )
pathOut  = os.path.join( basepath , "data/cons/" )
assert(os.path.exists(pathInp))
if not os.path.exists(pathOut):
	os.makedirs(pathOut)

mask_full  = xr.open_dataset(os.path.join( pathInp , "land_sea_mask_IPCC_antarctica.nc"), mask_and_scale=False)
mask = mask_full.land_sea_mask.astype(int)
lat = mask.lat
lon = mask.lon

#ind_coords = np.argwhere(mask.where(mask>0,0).values)
#ind_lat = ind_coords[:,0]
#ind_lon = ind_coords[:,1]

#icoord = 1460
#idx_lat = ind_lat[icoord]
#idx_lon = ind_lon[icoord]

#pt_lat = lat[idx_lat]
#pt_lon = lon[idx_lon]

## Some global parameters
##=======================
time_period    = np.arange( 1850 , 2101 , 1 , dtype = int )
time_reference = np.arange( 1961 , 1991 , 1 , dtype = int )
bayes_kwargs = { "n_mcmc_drawn_min" : 2500 , "n_mcmc_drawn_max" : 5000 }
n_sample    = 1000
ns_law      = nsm.GEV()
verbose     = "--not-verbose"
ci          = 0.05

# Niveau de confiance souhaité
CONF = 90

# Bornes
qInf = (100 - CONF) / 2.
qSup = 100 - qInf

# Return periods
rp_tmp = np.linspace(2, 100, 99)
rp = rp_tmp.astype(int)
nrp = len(rp)

## Shiny app UI
##=======================
app_ui = ui.page_sidebar(
	
	ui.sidebar(
		"Input parameters",
		ui.input_selectize("var_nx", "Variable", choices=variables, selected="maximum temperature"),
		ui.input_selectize("duration", "Duration of the event in days", choices=durations, selected="3"),
		ui.input_slider("year","Year",min = min(year),max = max(year),value = 2024,step = 1,sep=""),
		ui.input_numeric("coord_lat", "Latitude:", lat1,min=-89.5,max=89.5),
		ui.input_numeric("coord_lon", "Longitude:", lon1,min=-179.5,max=179.5),
	), 
	ui.layout_columns(
		ui.card(
            ui.p("Click on the marker and drag at the desired location. Calculations are available on continental areas only (except Antarctica)."),
			output_widget("map"),
			ui.output_plot("plot_ts")
		),
		ui.navset_card_tab(
			ui.nav_panel(
				"Attribution of a new event",
				"Under this heading, you determine whether or not a newly observed event is attributable to climate change. Just specify its location, duration in days, and intensity in degrees Celsius. The slider on the left can be used to determine the properties of the event year by year.",
				ui.input_numeric("user_value", "Temperature in °C to be attributed:", f""),
				ui.input_task_button("go_forecast", "Compute probabilities (takes 1 minute)", class_="btn-success"),
				ui.output_data_frame("proba_df"),
				ui.output_plot("plot_proba"),                   
				ui.output_plot("plot_far"),
				#ui.output_plot("plot_param")
			),
			ui.nav_menu(
				"Climate monitoring",
				ui.nav_panel(
					"Compute return level from a given return period",
					"Under this heading, you calculate the return level in degrees Celsius associated with a return period. The slider on the left can be used to determine the properties of the event year by year.",
					ui.input_slider("dr","Return Period (in years)",min = rp[0],max = rp[-1],value = 50,step = 1,sep=""),
					ui.input_task_button("go_dr", "Compute return level (takes 1 minute)", class_="btn-success"),
					ui.output_data_frame("rl_df"),
					ui.output_plot("plot_dr")                 
				),
				ui.nav_panel( 
					"Compute return period from a given return level",
					"Under this heading, you calculate the return period of an observed event in degrees Celsius. The slider on the left can be used to determine the properties of the event year by year.",
					ui.input_numeric("rl_val", "Return level (temperature in °C):", f""),
					ui.input_task_button("go_rl", "Compute return period (takes 1 minute)", class_="btn-success"),
					ui.output_data_frame("rp_df"),
					ui.output_plot("plot_rl"),
					#ui.output_plot("plot_param_rl")
				),
			)
		),
		col_widths=(6, 6)
	),
    ui.p("Reference: Qasmi et al. 2025, submitted: An automatic procedure for the attribution of climate extremes events at the global scale"),	
	title="Extreme event fast attribution application",
	
)
	

	


def server(input, output, session):

	set_cmdstan_path(os.path.join('/d0/www-ubuntu-extreme-event-app2025/.cmdstan/cmdstan-2.36.0/'))

	# Reactive values to store location information
	loc1 = reactive.value()

	# Update the reactive values when the selectize inputs change
	@reactive.effect
	def _():
		loc1.set(loc_str_to_coords(input.coord_lat(),input.coord_lon()))
			
	# When a marker is moved, the input value gets updated to "lat, lon",
	# so we decode that into a dict
	def loc_str_to_coords(x_lat: str, x_lon: str) -> dict:

		#if (x_lat.isnumeric() and x_lon.isnumeric()):
		#    lat = float(x_lat)
		#    lon = float(x_lon)
		#    return {"latitude": lat, "longitude": lon}
		#else:
		#    raise SafeException("This is a safe exception")
		#    return {}
		lat = x_lat
		lon = x_lon
		return {"latitude": lat, "longitude": lon}

	# Convenient way to get the lat/lons as a tuple
	@reactive.calc
	def loc1xy():
		return loc1()["latitude"], loc1()["longitude"]

	# For performance, render the map once and then perform partial updates
	# via reactive side-effects
	@render_widget
	def map():
		map_tas = L.Map(basemap=L.basemaps.CartoDB.Positron, 
						center=center, 
						zoom=zoom, 
						scroll_wheel_zoom = True, 
						world_copy_jump = True, 
						bounce_at_zoom_limits = False, 
						fit_bounds=[[-90, 0], [90, 360]], 
						min_zoom=3, 
						max_zoom=6,
						)
		
		return map_tas #L.Map(zoom=6, center=center)

	# Add marker for first location
	@reactive.effect
	def _():
		update_marker(map.widget, loc1xy(), on_move1, "coordinates")


	# ---------------------------------------------------------------
	# Helper functions
	# ---------------------------------------------------------------
	def update_marker(map: L.Map, loc: tuple, on_move: object, name: str):
		remove_layer(map, name)
		m = L.Marker(location=loc, draggable=True, name=name)
		m.on_move(on_move)
		map.add_layer(m)

	def remove_layer(map: L.Map, name: str):
		for layer in map.layers:
			if layer.name == name:
				map.remove_layer(layer)

	def on_move1(**kwargs):
		return on_move("coordinates", **kwargs)

	# When the markers are moved, update the selectize inputs to include the new
	# location (which results in the locations() reactive value getting updated,
	# which invalidates any downstream reactivity that depends on it)
	def on_move(id, **kwargs):
		loc = kwargs["location"]
		lat_str = f"{loc[0]}"
		lon_str = f"{loc[1]}"
		ui.update_text("coord_lat", value=lat_str)
		ui.update_text("coord_lon", value=lon_str)
	 
	@reactive.calc
	@reactive.event(input.go_forecast, ignore_none=True, ignore_init=True)
	def cons_calc():

		set_seed: np.random.seed(1)

		input.go_forecast()

		with reactive.isolate():
			#year_user = input.year()
			#var_user = input.variable()
			val_abs = input.user_value()
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()
        
		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		#event = ns.Event( "lat"+str(lat_sub)+"-lon"+str(lon_sub) , time=2035 , reference=time_reference , type_ = "value" , variable = var_user, unit = "K" )
		#print(event)

		## Load models and observations
		##=============================
		Xo,Yo_tmp,lat_real,lon_real = load_obs( pathInp , lat_sub, lon_sub )
		
		if not val_abs:
			raise Exception("A numeric value is required")
		
		val_max = np.mean(Yo_tmp.values-273.15) + 4*np.std(Yo_tmp.values)
		val_min = np.min(Yo_tmp.values-273.15)
		
		if val_abs > val_max or val_abs < val_min:
			raise Exception("A realistic value is required")

		## Anomaly from observations
		##==========================
		new_Yo = pd.DataFrame(data=[val_abs+273.15], index=[dt.date.today().year])
		Yo_tmp = Yo_tmp._append(new_Yo)

		bias = { "Multi_Synthesis" : Yo_tmp.loc[time_reference].mean() }
		Yo = Yo_tmp - bias["Multi_Synthesis"]

		#event.value = float(Yo.loc[event.time])

		## Models in celsius
		##==================
		Xo -= Xo.loc[time_reference].mean()
		Yo -= Yo.loc[time_reference].mean()

		## Multi-model
		##============
		climMM_file = os.path.join( pathInp , "climMM/climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout

		#np.bool=np.bool_
		#climCXCB   = ns.constrain_law( climCX , Yo , verbose = verbose , **bayes_kwargs )
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', install_dir="/d0/www-ubuntu-extreme-event-app2025/.cmdstan/cmdstan-2.36.0", **bayes_kwargs)

		## Stats
		upper_side = "upper"
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC
		n_x=len(climCXCB.X.sample)-1 #N tirages de X
		n_ess=int((len(climCXCB.law_coef.sample_MCMC)-1)/n_x) # N de tirages par chaine

		n_stat = 6

		event_user = val_abs+273.15-bias["Multi_Synthesis"].values

		## Output
		stats = xr.DataArray( np.zeros( (ny,nsample_MCMC,n_stat) ) , coords = [climCXCB.X.time , samples_MCMC , ["pC","pF","IC","IF","PR","dI"] ] , dims = ["time","sample_MCMC","stats"] )

		XF_noBE=np.tile(climCXCB.X.loc[:,:,"F","Multi_Synthesis"][:,1:], (1, n_ess)).T
		XF_data=np.vstack((XF_noBE,climCXCB.X.loc[:,"BE","F","Multi_Synthesis"])) #Ne pas répéter le BE.
		XF=xr.DataArray( XF_data.T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )

		XC_noBE=np.tile(climCXCB.X.loc[:,:,"C","Multi_Synthesis"][:,1:], (1, n_ess)).T
		XC_data=np.vstack((XC_noBE,climCXCB.X.loc[:,"BE","C","Multi_Synthesis"])) #Ne pas répéter le BE.
		XC=xr.DataArray( XC_data.T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )

		#XF = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","F","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		#XC = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","C","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		
		locF  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		locC  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		scaleF = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		scaleC = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		shape = climCXCB.law_coef.loc["shape0",:,"Multi_Synthesis"] + xr.zeros_like(locF)

		stats.loc[:,:,"pF"] = sc.genextreme.sf( event_user , loc = locF , scale = scaleF , c = - shape ).T
		stats.loc[:,:,"pC"] = sc.genextreme.sf( event_user , loc = locC , scale = scaleC , c = - shape ).T

		## PR
		stats.loc[:,:,"PR"] = stats.loc[:,:,"pF"] / stats.loc[:,:,"pC"]

		## deltaI
		stats.loc[:,:,"dI"] = stats.loc[:,:,"IF"] - stats.loc[:,:,"IC"]

		return stats, val_abs, lat_real, lon_real, Yo_tmp, climMM, climCXCB

	@render.plot
	def plot_ts():  

		if not input.coord_lat():
			raise SafeException("Latitude is a numeric value between -90 and 90")
		if not input.coord_lat() or not input.coord_lon():
			raise SafeException("Longitude is a numeric value between -180 and 180")

		#with reactive.isolate():
			#year_user = input.year()
			#var_user = input.variable()
		lat_sub = input.coord_lat()
		lon_sub = input.coord_lon()

		## Load models and observations
		##=============================
		Xo,Yo,lat_real,lon_real = load_obs( pathInp , lat_sub, lon_sub )

		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		#pdf = mpdf.PdfPages( "era5_lat"+str(round(lat_sub,2))+"_lon"+str(round(lon_sub,2))+".pdf" )

		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( Yo-273.15 , color = "black" )
		ax.set_ylabel("°C")
		ax.set_xlabel("Year")
		ax.set_title("ERA5 annual "+input.duration()+"-day mean "+input.var_nx()+" at ["+str(lat_real)+"°N ; "+str(lon_real)+" °E]")
		plt.tight_layout()

		#pdf.savefig( fig )
		#plt.close(fig)
		#pdf.close()

		return fig
	
	@render.data_frame  
	def proba_df():
		
		stats = cons_calc()[0]
		ci = 0.05
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )
		qstats.loc["be",:,"PR"] = stats.loc[:,"BE","PR"]
		
		prop_PR_not_valid = ( ~(stats.loc[:,:,"pF"] > 0) & ~(stats.loc[:,:,"pC"] > 0) ).sum( dim = "sample_MCMC" ) / ( stats.sample_MCMC.size - 1 )
		PR_not_valid = prop_PR_not_valid > ci

		qstats.loc["ql",PR_not_valid,"PR"] = 0
		qstats.loc["be",PR_not_valid,"PR"] = 1
		qstats.loc["qu",PR_not_valid,"PR"] = np.inf

		qstats.loc["ql",~np.isfinite(qstats.loc["ql",:,"PR"]),"PR"] = 0
		#qstats.loc["be",~np.isfinite(qstats.loc["be",:,"PR"]),"PR"] = 1
		qstats.loc["qu",~np.isfinite(qstats.loc["qu",:,"PR"]),"PR"] = np.inf

		df = pd.DataFrame({"Probability for "+str(input.year()) : ["pF","pC","pR"] , "Q05": qstats.loc["ql",input.year(),["pF","pC","PR"]], "Best estimate": qstats.loc["be",input.year(),["pF","pC","PR"]], "Q95": qstats.loc["qu",input.year(),["pF","pC","PR"]]})
		df = df.replace([np.inf], ["Inf"])
		return render.DataGrid(df)  
	
	@render.plot
	def plot_proba():

		stats = cons_calc()[0]
		ci = 0.05
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		colors = ["red","blue"]
		yticks = np.array([0,1e-12,1e-6,1e-3,1e-2,1/30,1/10,0.2,0.5,1] )
		yticklabelsL = ["0",r"$10^{-12}$",r"$10^{-6}$",r"$10^{-3}$",r"$10^{-2}$","1/30","1/10","1/5","1/2","1"]
		yticklabelsR = [r"$\infty$","","","1000","100","30","10","5","2","1"]


		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot( nrow , ncol , 1 )
		for iqp,qp in enumerate([qstats.loc[:,:,"pF"],qstats.loc[:,:,"pC"]]):
			ax.plot( qp.time , plink(qp.loc["be",:]) , color = colors[iqp] )
			ax.fill_between( qp.time , plink(qp.loc["ql",:]) , plink(qp.loc["qu",:]) , color = colors[iqp] , alpha = 0.5 )
		ax.set_yticks(plink(yticks))
		ax.set_yticklabels(yticklabelsL)
		ax.set_ylabel("Probability")
		ax.set_ylim(plink([0,1]))

		axR = fig.add_subplot( nrow , ncol , 1 , sharex = ax , frameon = False )
		axR.yaxis.tick_right()
		axR.set_yticks(plink(yticks))
		axR.set_yticklabels(yticklabelsR)
		axR.yaxis.set_label_position( "right" )
		axR.set_ylabel( "Return period" , rotation = 270 )
		ax.set_ylim(plink([0,1]))

		ax.set_title("Probabilities for an event of "+str(cons_calc()[1])+"°C at "+str(cons_calc()[2])+"°N / "+str(cons_calc()[3])+" °E")
		label = ["Factual", "Counterfactual"]
		legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["red","blue"],label)]
		ax.legend( handles = legend , loc = "upper left" )


		plt.tight_layout()

		return fig

	@render.plot
	def plot_far():

		stats = cons_calc()[0]
		ci = 0.05
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )
		qstats.loc["be",:,"PR"] = stats.loc[:,"BE","PR"]
		
		prop_PR_not_valid = ( ~(stats.loc[:,:,"pF"] > 0) & ~(stats.loc[:,:,"pC"] > 0) ).sum( dim = "sample_MCMC" ) / ( stats.sample_MCMC.size - 1 )
		PR_not_valid = prop_PR_not_valid > ci

		qstats.loc["ql",PR_not_valid,"PR"] = 0
		qstats.loc["be",PR_not_valid,"PR"] = 1
		qstats.loc["qu",PR_not_valid,"PR"] = np.inf

		qstats.loc["ql",~np.isfinite(qstats.loc["ql",:,"PR"]),"PR"] = 0
		#qstats.loc["be",~np.isfinite(qstats.loc["be",:,"PR"]),"PR"] = 1
		qstats.loc["qu",~np.isfinite(qstats.loc["qu",:,"PR"]),"PR"] = np.inf
		
		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		yticks       = np.array([0 , 1e-3 , 1e-2 , 1e-1 , 1 , 5 , 10 , 100 , 1000 , np.inf])
		yticklabelsL = np.array(["0",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$","1","5","10","100","1000",r"$\infty$"])
		yticklabelsR = np.array([r"$-\infty$", "-999" , "-99","-9","0","0.8","0.9","0.99","0.999","1"])

		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot(1,1,1)
		ax.plot( qstats.time , PRlink(qstats.loc["be",:,"PR"]) , color = "red" )
		ax.fill_between( qstats.time , PRlink(qstats.loc["ql",:,"PR"]) , PRlink(qstats.loc["qu",:,"PR"]) , color = "red" , alpha = 0.5 )
		ax.set_yticks(PRlink(yticks))
		ax.set_yticklabels(yticklabelsL)
		ax.set_ylabel("Probability ratio")
		ax.set_ylim(PRlink([0,np.inf]))

		axR = fig.add_subplot( 1 , 1 , 1 , sharex = ax , frameon = False )
		axR.yaxis.tick_right()
		axR.set_yticks(PRlink(yticks))
		axR.set_yticklabels(yticklabelsR)
		axR.yaxis.set_label_position( "right" )
		axR.set_ylabel( "FAR" , rotation = 270 )
		axR.set_ylim(PRlink([0,np.inf]))

		plt.tight_layout()
		return fig

	@render.plot
	def plot_param():

		climMM = cons_calc()[5]
		climCXCB = cons_calc()[6]

		ci=0.05
		qcoefc = climCXCB.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )
		qcoef  = climMM.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )

		qcoefc.loc[["ql","qu"],:,:] = qcoefc.loc[["ql","qu"],:,:] - qcoefc.loc["BE",:,:]
		qcoef = qcoef - qcoefc.loc["BE",:,:]

		## mpl parameter
		ymin = min( (climCXCB.law_coef - qcoefc.loc["BE",:,:]).min() , (climMM.law_coef - qcoefc.loc["BE",:,:]).min() )
		ymax = max( (climCXCB.law_coef - qcoefc.loc["BE",:,:]).max() , (climMM.law_coef - qcoefc.loc["BE",:,:]).max() )
		delta = 0.1 * (ymax-ymin)
		ylim = (ymin-delta,ymax+delta)

		label = ["Prior distribution", "Posterior"]
		legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["pink","red"],label)]
		#	legend.append( mplpatch.Patch(facecolor = "red"  , edgecolor = "red"  , label = "clim"       , alpha = 0.5 ) )
		#	legend.append( mplpatch.Patch(facecolor = "blue" , edgecolor = "blue" , label = "clim_const" , alpha = 0.5 ) )


		kwargs = { "positions" : range(climCXCB.n_coef) , "showmeans" : False , "showextrema" : False , "showmedians" : False }


		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot(1,1,1)

		## violin plot
		vplotc = ax.violinplot( (climCXCB.law_coef - qcoefc.loc["BE",:,:])[:,1:,:].loc[:,:,"Multi_Synthesis"].values.T , **kwargs )
		vplot  = ax.violinplot( (climMM.law_coef - qcoefc.loc["BE",:,:])[:,1:,:].loc[:,:,"Multi_Synthesis"].values.T , **kwargs )

		## Change color
		for pc in vplotc["bodies"]:
			pc.set_facecolor("red")
			pc.set_edgecolor("red")
			pc.set_alpha(0.5)

		for pc in vplot["bodies"]:
			pc.set_facecolor("pink")
			pc.set_edgecolor("pink")
			pc.set_alpha(0.4)

		# add quantiles
		for i in range(climCXCB.n_coef):
			for q in ["ql","qu"]:
				ax.hlines( qcoefc[:,i,:].loc[q,"Multi_Synthesis"] , i - 0.3 , i + 0.3 , color = "red" )
			for q in ["ql","qu","BE"]:
				ax.hlines( qcoef[:,i,:].loc[q,"Multi_Synthesis"] , i - 0.3 , i + 0.3 , color = "pink" )

		ax.hlines( 0 , -0.5 , climCXCB.n_coef-0.5 , color = "black" )
		for i in range(climCXCB.n_coef-1):
			ax.vlines( i + 0.5 , ylim[0] , ylim[1] , color = "grey" )

		## some params
		ax.set_xlim((-0.5,climCXCB.n_coef-0.5))
		ax.set_xticks(range(climCXCB.n_coef))
		xticks = [ "{}".format(p) + "{}".format( "-" if np.sign(q) > 0 else "+" ) + r"${}$".format(float(np.sign(q)) * round(float(q),2)) for p,q in zip(climCXCB.ns_law.get_params_names(True),qcoefc.loc["BE",:,"Multi_Synthesis"]) ]
		ax.set_xticklabels( xticks )#, fontsize = 20 )
		#for item in ax.get_yticklabels():
	#		item.set_fontsize(20)
		ax.set_ylim(ylim)

		ax.set_title( "GEV parameters "  )
		ax.legend( handles = legend  )

		fig.set_tight_layout(True)

		return fig


	@reactive.calc
	@reactive.event(input.go_dr, ignore_none=True, ignore_init=True)
	def dr_calc():

		set_seed: np.random.seed(1)

		input.go_dr()

		with reactive.isolate():
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()
			p = input.dr()

		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		## Load models and observations
		##=============================
		Xo,Yo_tmp,lat_real,lon_real = load_obs( pathInp , lat_sub, lon_sub )

		## Anomaly from observations
		##==========================
		#bias = { "Multi_Synthesis" : Yo.loc[event.reference].mean() }
		#Yo -= bias["Multi_Synthesis"]

		bias = { "Multi_Synthesis" : Yo_tmp.loc[time_reference].mean() }
		Yo = Yo_tmp - bias["Multi_Synthesis"]

		#event.value = float(Yo.loc[event.time])

		## Models in celsius
		##==================
		Xo -= Xo.loc[time_reference].mean()
		Yo -= Yo.loc[time_reference].mean()

		## Multi-model
		##============
		climMM_file = os.path.join( pathInp , "climMM/climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', install_dir="/d0/www-ubuntu-extreme-event-app2025/.cmdstan/cmdstan-2.36.0", **bayes_kwargs)

		## Stats
		upper_side = "upper"
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC

		## Output
		n_stat = 3
		stats = xr.DataArray( np.zeros( (ny,nsample_MCMC,n_stat) ) , coords = [climCXCB.X.time , samples_MCMC , ["IF","IC","dI"] ] , dims = ["time","sample_MCMC","stats"] )

		XF = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","F","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		XC = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","C","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		
		locF  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		locC  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		scaleF = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		scaleC = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		shape = climCXCB.law_coef.loc["shape0",:,"Multi_Synthesis"] + xr.zeros_like(locF)

		p_user = p + xr.zeros_like(locF)
		bias_md = xr.DataArray( np.tile(np.repeat(bias["Multi_Synthesis"].values - 273.15,ny), (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		stats.loc[:,:,"IF"] = sc.genextreme.isf( 1/p_user , loc = locF , scale = scaleF , c = - shape ).T + bias_md
		stats.loc[:,:,"IC"] = sc.genextreme.isf( 1/p_user , loc = locC , scale = scaleC , c = - shape ).T + bias_md
		## deltaI
		stats.loc[:,:,"dI"] = stats.loc[:,:,"IF"] - stats.loc[:,:,"IC"]

		return stats, lat_real, lon_real
	
	@render.data_frame  
	def rl_df():

		stats = dr_calc()[0]
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		df = pd.DataFrame({"Return level estimated in "+str(input.year()) : ["Factual" , "Counterfactual", "Difference"] , "Q05": qstats.loc["ql",input.year(),["IF","IC","dI"]], "Best estimate": qstats.loc["be",input.year(),["IF","IC","dI"]], "Q95": qstats.loc["qu",input.year(),["IF","IC","dI"]]})
		return render.DataGrid(df)  

	@render.plot
	def plot_dr():

		stats = dr_calc()[0]
		ci = 0.05
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 2
		width  = 210*mm
		height = width / ncol / ratio * nrow

		fig = plt.figure( figsize = (width,height) )

		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( qstats.time , qstats.loc["be",:,"IC"], color = "blue" )
		ax.plot( qstats.time , qstats.loc["be",:,"IF"] , color = "red"  )
		ax.fill_between( qstats.time , qstats.loc["ql",:,"IC"] , qstats.loc["qu",:,"IC"] , color = "blue" , alpha = 0.5 )
		ax.fill_between( qstats.time , qstats.loc["ql",:,"IF"] , qstats.loc["qu",:,"IF"] , color = "red"  , alpha = 0.5 )
		ax.set_ylabel("Return level")
		
		ax.set_title("Return level of a "+str(input.dr())+"-year event at ["+str(dr_calc()[1])+"°N ; "+str(dr_calc()[2])+" °E]")

		#ax = fig.add_subplot( nrow , ncol , 2 )
		#ax.plot( qdI.time , qdI.loc["BE",:] , color = "red" )
		#ax.fill_between( qdI.time , qdI.loc["QL",:] , qdI.loc["QU",:] , color = "red"  , alpha = 0.5 )

		plt.tight_layout()


		return fig

	@reactive.calc
	@reactive.event(input.go_rl, ignore_none=True, ignore_init=True)
	def rl_calc():

		set_seed: np.random.seed(1)

		input.go_rl()

		with reactive.isolate():
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()
			rl_user = input.rl_val()+273.15

		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		## Load models and observations
		##=============================
		Xo,Yo_tmp,lat_real,lon_real = load_obs( pathInp , lat_sub, lon_sub )

		if not input.rl_val():
			raise Exception("A numeric value is required")

		val_max = np.mean(Yo_tmp.values) + 4*np.std(Yo_tmp.values)
		val_min = np.min(Yo_tmp.values)
		
		if rl_user > val_max or rl_user < val_min:
			raise Exception("A realistic value is required")

		## Anomaly from observations
		##==========================
		#bias = { "Multi_Synthesis" : Yo.loc[event.reference].mean() }
		#Yo -= bias["Multi_Synthesis"]

		bias = { "Multi_Synthesis" : Yo_tmp.loc[time_reference].mean() }
		Yo = Yo_tmp - bias["Multi_Synthesis"]

		#event.value = float(Yo.loc[event.time])

		## Models in celsius
		##==================
		Xo -= Xo.loc[time_reference].mean()
		Yo -= Yo.loc[time_reference].mean()

		## Multi-model
		##============
		climMM_file = os.path.join( pathInp , "climMM/climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', install_dir="/d0/www-ubuntu-extreme-event-app2025/.cmdstan/cmdstan-2.36.0", **bayes_kwargs)

		## Stats
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC

		rl = rl_user-bias["Multi_Synthesis"].values

		## Output
		n_stat = 3
		stats = xr.DataArray( np.zeros( (ny,nsample_MCMC,n_stat) ) , coords = [climCXCB.X.time , samples_MCMC , ["pC","pF", "PR"] ] , dims = ["time","sample_MCMC","stats"] )
		#stats_pr = xr.DataArray( np.zeros( (ny,nsample_MCMC) ) , coords = [climCXCB.X.time , samples_MCMC , ["pC","pF", "PR"] ] , dims = ["time","sample_MCMC","stats"] )

		XF = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","F","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		XC = xr.DataArray( np.tile(climCXCB.X.loc[:,"BE","C","Multi_Synthesis"], (nsample_MCMC, 1)).T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )
		
		locF  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		locC  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
		scaleF = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		scaleC = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XC * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
		shape = climCXCB.law_coef.loc["shape0",:,"Multi_Synthesis"] + xr.zeros_like(locF)

		stats.loc[:,:,"pF"] = sc.genextreme.sf( rl , loc = locF , scale = scaleF , c = - shape ).T
		stats.loc[:,:,"pC"] = sc.genextreme.sf( rl , loc = locC , scale = scaleC , c = - shape ).T
		## PR
		stats_pr = stats.loc[:,:,"pF"] / stats.loc[:,:,"pC"]

		return stats, stats_pr, lat_real, lon_real, climMM, climCXCB
	
	@render.data_frame  
	def rp_df():

		stats = 1/rl_calc()[0]
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		stats_pr = rl_calc()[1]
		qstats_pr = stats_pr.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		qstats.loc[:,:,"PR"] = qstats_pr
		qstats.loc["be",:,"PR"] = stats_pr.loc[:,"BE"]
		
		prop_PR_not_valid = ( ~(stats.loc[:,:,"pF"] > 0) & ~(stats.loc[:,:,"pC"] > 0) ).sum( dim = "sample_MCMC" ) / ( stats.sample_MCMC.size - 1 )
		PR_not_valid = prop_PR_not_valid > ci

		qstats.loc["ql",PR_not_valid,"PR"] = 0
		qstats.loc["be",PR_not_valid,"PR"] = 1
		qstats.loc["qu",PR_not_valid,"PR"] = np.inf

		qstats.loc["ql",~np.isfinite(qstats.loc["ql",:,"PR"]),"PR"] = 0
		#qstats.loc["be",~np.isfinite(qstats.loc["be",:,"PR"]),"PR"] = 1
		qstats.loc["qu",~np.isfinite(qstats.loc["qu",:,"PR"]),"PR"] = np.inf

		#df = pd.DataFrame({"Probability for "+str(input.year()) : ["pF","pC","pR"] , "Q05": qstats.loc["ql",input.year(),["pF","pC","PR"]], "Best estimate": qstats.loc["be",input.year(),["pF","pC","PR"]], "Q95": qstats.loc["qu",input.year(),["pF","pC","PR"]]})

		df = pd.DataFrame({"Return period estimated in "+str(input.year()) : ["Factual" , "Counterfactual", "Ratio"] , "Q05": qstats.loc["ql",input.year(),["pF","pC","PR"]], "Best estimate": qstats.loc["be",input.year(),["pF","pC","PR"]], "Q95": qstats.loc["qu",input.year(),["pF","pC","PR"]]})
		df = df.replace([np.inf], ["Inf"])
		return render.DataGrid(df)  
		
	@render.plot
	def plot_rl():

		stats = rl_calc()[0]
		ci = 0.05
		qstats = stats.quantile( [ci / 2 ,  0.5 , 1 - ci / 2 ] , dim = "sample_MCMC" ).assign_coords( quantile = [ "ql" , "be" , "qu" ] )

		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		colors = ["red","blue"]
		yticks = np.array([0,1e-12,1e-6,1e-3,1e-2,1/30,1/10,0.2,0.5,1] )
		yticklabelsL = ["0",r"$10^{-12}$",r"$10^{-6}$",r"$10^{-3}$",r"$10^{-2}$","1/30","1/10","1/5","1/2","1"]
		yticklabelsR = [r"$\infty$","","","1000","100","30","10","5","2","1"]

		#pdf = mpdf.PdfPages( "return_period_lat"+str(round(rl_calc()[2],2))+"_lon"+str(round(rl_calc()[3],2))+".pdf" )

		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot( nrow , ncol , 1 )
		for iqp,qp in enumerate([qstats.loc[:,:,"pF"],qstats.loc[:,:,"pC"]]):
			ax.plot( qp.time , plink(qp.loc["be",:]) , color = colors[iqp] )
			ax.fill_between( qp.time , plink(qp.loc["ql",:]) , plink(qp.loc["qu",:]) , color = colors[iqp] , alpha = 0.5 )
		ax.set_yticks(plink(yticks))
		ax.set_yticklabels(yticklabelsL)
		ax.set_ylabel("Probability")
		ax.set_ylim(plink([0,1]))

		axR = fig.add_subplot( nrow , ncol , 1 , sharex = ax , frameon = False )
		axR.yaxis.tick_right()
		axR.set_yticks(plink(yticks))
		axR.set_yticklabels(yticklabelsR)
		axR.yaxis.set_label_position( "right" )
		axR.set_ylabel( "Return period" , rotation = 270 )
		ax.set_ylim(plink([0,1]))
		ax.set_title("Return period of a "+str(input.rl_val())+"°C-event at ["+str(rl_calc()[2])+"°N ; "+str(rl_calc()[3])+" °E]")
		label = ["Factual", "Counterfactual"]
		legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["red","blue"],label)]
		ax.legend( handles = legend , loc = "upper left" )

		plt.tight_layout()

		#pdf.savefig( fig )
		#plt.close(fig)
		#pdf.close()

		return fig

	@render.plot
	def plot_param_rl():

		climMM = rl_calc()[4]
		climCXCB = rl_calc()[5]

		#pdf = mpdf.PdfPages( "law_coef_lat"+str(round(rl_calc()[2],2))+"_lon"+str(round(rl_calc()[3],2))+".pdf" )

		ci=0.05
		qcoefc = climCXCB.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )
		qcoef  = climMM.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )

		qcoefc.loc[["ql","qu"],:,:] = qcoefc.loc[["ql","qu"],:,:] - qcoefc.loc["BE",:,:]
		qcoef = qcoef - qcoefc.loc["BE",:,:]

		## mpl parameter
		ymin = min( (climCXCB.law_coef - qcoefc.loc["BE",:,:]).min() , (climMM.law_coef - qcoefc.loc["BE",:,:]).min() )
		ymax = max( (climCXCB.law_coef - qcoefc.loc["BE",:,:]).max() , (climMM.law_coef - qcoefc.loc["BE",:,:]).max() )
		delta = 0.1 * (ymax-ymin)
		ylim = (ymin-delta,ymax+delta)

		label = ["Prior distribution", "Posterior"]
		legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["pink","red"],label)]
		#	legend.append( mplpatch.Patch(facecolor = "red"  , edgecolor = "red"  , label = "clim"       , alpha = 0.5 ) )
		#	legend.append( mplpatch.Patch(facecolor = "blue" , edgecolor = "blue" , label = "clim_const" , alpha = 0.5 ) )


		kwargs = { "positions" : range(climCXCB.n_coef) , "showmeans" : False , "showextrema" : False , "showmedians" : False }


		mm     = 1. / 25.4
		ratio  = 16 / 11
		nrow   = 1
		ncol   = 1
		width  = 180*mm
		height = width / ncol / ratio * nrow

		fig = plt.figure( figsize = (width,height) )
		ax  = fig.add_subplot(1,1,1)

		## violin plot
		vplotc = ax.violinplot( (climCXCB.law_coef - qcoefc.loc["BE",:,:])[:,1:,:].loc[:,:,"Multi_Synthesis"].values.T , **kwargs )
		vplot  = ax.violinplot( (climMM.law_coef - qcoefc.loc["BE",:,:])[:,1:,:].loc[:,:,"Multi_Synthesis"].values.T , **kwargs )

		## Change color
		for pc in vplotc["bodies"]:
			pc.set_facecolor("red")
			pc.set_edgecolor("red")
			pc.set_alpha(0.5)

		for pc in vplot["bodies"]:
			pc.set_facecolor("pink")
			pc.set_edgecolor("pink")
			pc.set_alpha(0.4)

		# add quantiles
		for i in range(climCXCB.n_coef):
			for q in ["ql","qu"]:
				ax.hlines( qcoefc[:,i,:].loc[q,"Multi_Synthesis"] , i - 0.3 , i + 0.3 , color = "red" )
			for q in ["ql","qu","BE"]:
				ax.hlines( qcoef[:,i,:].loc[q,"Multi_Synthesis"] , i - 0.3 , i + 0.3 , color = "pink" )

		ax.hlines( 0 , -0.5 , climCXCB.n_coef-0.5 , color = "black" )
		for i in range(climCXCB.n_coef-1):
			ax.vlines( i + 0.5 , ylim[0] , ylim[1] , color = "grey" )

		## some params
		ax.set_xlim((-0.5,climCXCB.n_coef-0.5))
		ax.set_xticks(range(climCXCB.n_coef))
		xticks = [ "{}".format(p) + "{}".format( "-" if np.sign(q) > 0 else "+" ) + r"${}$".format(float(np.sign(q)) * round(float(q),2)) for p,q in zip(climCXCB.ns_law.get_params_names(True),qcoefc.loc["BE",:,"Multi_Synthesis"]) ]
		ax.set_xticklabels( xticks )#, fontsize = 20 )
		#for item in ax.get_yticklabels():
	#		item.set_fontsize(20)
		ax.set_ylim(ylim)

		ax.set_title( "GEV parameters "  )
		ax.legend( handles = legend  )

		fig.set_tight_layout(True)

		#pdf.savefig( fig )
		#plt.close(fig)
		#pdf.close()

		return fig

app = App(app_ui, server)
