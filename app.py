import matplotlib.pyplot as plt
import ipyleaflet as L
import ipywidgets as widgets
from branca.colormap import linear
import xarray as xr
import numpy as np
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  
from shiny.types import SafeException
from shiny.types import SilentException

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

## Plot libraries
##===============
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mplpatch

# SDFC a tendance a afficher plein de warnings pendant les fits : on les désactive
import warnings
warnings.simplefilter("ignore")

################################################
## Fonctions calcul ##
################################################

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
		
	return Xo,Yo
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
lat1 = 48 #43.6
lon1 = 26 #1.43
center = [lat1, lon1]

variables = ["Max temperature"]
durations = ["3"]

## Path
##=====
basepath = os.getcwd() 
pathInp  = os.path.join( basepath , "data/"  )
pathOut  = os.path.join( basepath , "data/cons" )
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
		ui.input_selectize("var_nx", "Variable", choices=variables, selected="Max temperature"),
		ui.input_selectize("duration", "Duration of the event in days", choices=durations, selected="3"),
		ui.input_slider("year","Year",min = min(year),max = max(year),value = 2024,step = 1,sep=""),
		ui.input_numeric("coord_lat", "Latitude:", lat1,min=-89.5,max=89.5),
		ui.input_numeric("coord_lon", "Longitude:", lon1,min=-179.5,max=179.5),
	),  
	ui.layout_columns(
		ui.card(
			output_widget("map"),
			output_widget("plot_ts")
		),
		ui.navset_card_tab(
			ui.nav_panel(
				"Event attribution",
				"Under this heading, you determine whether or not an event observed (or forecast) in 2024 is attributable to climate change. Just specify its location, duration in days, and intensity in degrees Celsius.",
				ui.input_numeric("user_value", "Temperature in °C to be attributed:", f""),
				ui.input_task_button("go_forecast", "Compute probabilities", class_="btn-success"),
				output_widget("plot_proba"),                        
				output_widget("plot_far")
			),
			ui.nav_panel(
				"Return level",
				"Under this heading, you calculate the return level in degrees Celsius associated with a return period.",
				ui.input_slider("dr","Return Period (in years)",min = rp[0],max = rp[-1],value = 50,step = 1,sep=""),
				ui.input_task_button("go_dr", "Compute return level", class_="btn-success"),
				output_widget("plot_dr"),                        
			),
			ui.nav_panel( 
				"Return period",
				"Under this heading, you calculate the return period of an event observed in degrees Celsius.",
				ui.input_numeric("rl_val", "Return level (temperature in °C):", f""),
				ui.input_task_button("go_rl", "Compute return period", class_="btn-success"),
				output_widget("plot_rl")
			),
		),
		col_widths=(4, 8)
	),
)
	

	


def server(input, output, session):

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

		input.go_forecast()

		with reactive.isolate():
			#year_user = input.year()
			#var_user = input.variable()
			val_abs = input.user_value()
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()


		if not input.user_value():
			raise Exception("A numeric value is required")

		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		#event = ns.Event( "lat"+str(lat_sub)+"-lon"+str(lon_sub) , time=2035 , reference=time_reference , type_ = "value" , variable = var_user, unit = "K" )
		#print(event)

		## Load models and observations
		##=============================
		Xo,Yo_tmp = load_obs( pathInp , lat_sub, lon_sub )

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
		climMM_file = os.path.join( pathInp , "climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout

		#np.bool=np.bool_
		#climCXCB   = ns.constrain_law( climCX , Yo , verbose = verbose , **bayes_kwargs )
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', **bayes_kwargs)

		## Stats
		upper_side = "upper"
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC
		
		n_stat = 6

		event_user = val_abs+273.15-bias["Multi_Synthesis"].values

		## Output
		stats = xr.DataArray( np.zeros( (ny,nsample_MCMC,n_stat) ) , coords = [climCXCB.X.time , samples_MCMC , ["pC","pF","IC","IF","PR","dI"] ] , dims = ["time","sample_MCMC","stats"] )

		## 
		law = climCXCB.ns_law

		for s_MCMC in samples_MCMC:
			
			s_X = str(s_MCMC.values).split("_")[0]

			## Start with params
			law.set_params( climCXCB.law_coef.loc[:,s_MCMC,"Multi_Synthesis"].values )

			## Go to factual world
			law.set_covariable( climCXCB.X.loc[:,s_X,"F","Multi_Synthesis"].values , time )

			## Find value of event definition
			value = np.zeros(ny) + event_user
			
			## Find pF
			stats.loc[:,s_MCMC,"pF"] = law.sf( value , time ) if upper_side else law.cdf( value , time )

			## Find probability of the event in factual world
			pF = np.zeros(ny) + stats.loc[time,s_MCMC,"pF"]

			## IF
			stats.loc[:,s_MCMC,"IF"] = law.isf( pF , time ) if upper_side else law.icdf( pF , time )

			## Find pC
			law.set_covariable( climCXCB.X.loc[:,s_X,"C","Multi_Synthesis"].values , time )
			stats.loc[:,s_MCMC,"pC"] = law.sf( value , time ) if upper_side else law.cdf( value , time )

			## IC
			stats.loc[:,s_MCMC,"IC"] = law.isf( pF , time ) if upper_side else law.icdf( pF , time )


		## PR
		stats.loc[:,:,"PR"] = stats.loc[:,:,"pF"] / stats.loc[:,:,"pC"]

		## deltaI
		stats.loc[:,:,"dI"] = stats.loc[:,:,"IF"] - stats.loc[:,:,"IC"]

		return stats, val_abs, lat_sub, lon_sub, Yo_tmp#, year_user

	@reactive.calc
	def p0_calc():
		df_pC = pd.DataFrame({"year": cons_calc()[0].time , "pC": cons_calc()[0].loc[:,'BE','pC']})
		df_pC = df_pC.set_index("year")
		# Sélection des quantiles
		vInf = np.percentile(cons_calc()[0], qInf, axis=1)
		vSup = np.percentile(cons_calc()[0], qSup, axis=1)
		df_pC["q%s" % int(qInf)] = vInf[:,0]
		df_pC["q%s" % int(qSup)] = vSup[:,0]

		return df_pC

	@reactive.calc
	def p1_calc():
		df_pF = pd.DataFrame({"year": cons_calc()[0].time , "pF": cons_calc()[0].loc[:,'BE','pF']})
		df_pF = df_pF.set_index("year")
		# Sélection des quantiles
		vInf = np.percentile(cons_calc()[0], qInf, axis=1)
		vSup = np.percentile(cons_calc()[0], qSup, axis=1)
		df_pF["q%s" % int(qInf)] = vInf[:,1]
		df_pF["q%s" % int(qSup)] = vSup[:,1]

		return df_pF

	@reactive.calc
	def pr_calc():
		df_PR = pd.DataFrame({"year": cons_calc()[0].time , "PR": cons_calc()[0].loc[:,'BE','PR']})
		df_PR = df_PR.set_index("year")
		# Sélection des quantiles
		vInf = np.percentile(cons_calc()[0], qInf, axis=1)
		vSup = np.percentile(cons_calc()[0], qSup, axis=1)
		df_PR["q%s" % int(qInf)] = vInf[:,4]
		df_PR["q%s" % int(qSup)] = vSup[:,4]

		return df_PR

	@render_widget
	def plot_ts():  

		if not input.coord_lat():
			raise SafeException("Latitude is a numeric value between -90 and 90")
		if not input.coord_lat() or not input.coord_lon():
			raise SafeException("Longitude is a numeric value between -180 and 180")

		input.go_forecast()

		#with reactive.isolate():
			#year_user = input.year()
			#var_user = input.variable()
		lat_sub = input.coord_lat()
		lon_sub = input.coord_lon()

		## Load models and observations
		##=============================
		Xo,Yo = load_obs( pathInp , lat_sub, lon_sub )

		fig_ts = go.Figure()

		fig_ts.add_trace(go.Scatter(
			x=Yo.index.values,
			y=Yo.values.ravel()-273.15,
			yhoverformat='.1f',
			mode='lines+markers',
			line=dict(color="black"),
			hoveron='points',
			name="Observations"
		))

		fig_ts.update_layout(title="Observed "+input.variable()+" at "+str(round(lat_sub,2))+"°N / "+str(round(lon_sub,2))+" °E")
		fig_ts.update_xaxes(title_text='year')
		fig_ts.update_yaxes(title_text='°C')

		return fig_ts

	@render_widget
	def plot_proba():

		fig_pb = go.Figure()

		fig_pb.add_trace(go.Scatter(
			x=np.concatenate([p0_calc().index.values, p0_calc().index.values[::-1]]),
			y=np.concatenate([p0_calc()["q95"], p0_calc()["q5"][::-1]]),
			yhoverformat='.2f',
			fill='toself',
			hoveron='points',
			line=dict(color="blue"),
			showlegend=False,
			name="Confidence interval (95%)"
		))

		fig_pb.add_trace(go.Scatter(
			x=np.concatenate([p1_calc().index.values, p1_calc().index.values[::-1]]),
			y=np.concatenate([p1_calc()["q95"], p1_calc()["q5"][::-1]]),
			yhoverformat='.2f',
			fill='toself',
			hoveron='points',
			line=dict(color="red"),
			showlegend=False,
			name="Confidence interval (95%)"
		))

		fig_pb.add_trace(go.Scatter(
			x=p0_calc().index.values,
			y=p0_calc()["pC"],
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="blue"),
			hoveron='points',
			name="Pre-industrial climate"
		))

		fig_pb.add_trace(go.Scatter(
			x=p1_calc().index.values,
			y=p1_calc()["pF"],
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="red"),
			hoveron='points',
			name="Historical+SSP5-8.5 CMIP6"
		))

		fig_pb.update_yaxes(title_text="Probability")
		fig_pb.update_layout(title="Probabilities for an event of "+str(cons_calc()[1])+"°C at "+str(round(cons_calc()[2],2))+"°N / "+str(round(cons_calc()[3],2))+" °E")

		return fig_pb

	@render_widget
	def plot_far():

		# FAR
		fig_far = go.Figure()
		fig_far = make_subplots(specs=[[{"secondary_y": True}]])

		fig_far.add_trace(go.Scatter(
			x=p1_calc().index.values,
			y=pr_calc()["PR"],
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="blue"),
			hoveron='points',
			name="Probability ratio"
		), secondary_y=False)

		FAR = 1 - 1/pr_calc()["PR"]

		fig_far.add_trace(go.Scatter(
			x=p0_calc().index.values,
			y=FAR,
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="green"),
			hoveron='points',
			name="FAR"
		), secondary_y=True)

		fig_far.update_yaxes(title_text="Probability ratio", secondary_y=False)
		fig_far.update_yaxes(title_text="FAR", secondary_y=True)
		fig_far.update_layout(title="Best estimates for FAR and PR for an event of "+str(cons_calc()[1])+"°C at "+str(round(cons_calc()[2],2))+"°N / "+str(round(cons_calc()[3],2))+" °E")

		return fig_far

	@reactive.effect
	def _():
		plot_proba.widget.layout.annotations=[]
		plot_proba.widget.layout.shapes=[]
		plot_proba.widget.add_vline(x = input.year(), line_width=3, line_dash="dot",annotation_text=str(input.year()), 
			annotation_position="bottom right",
			line_color="blue")
		plot_far.widget.layout.annotations=[]
		plot_far.widget.layout.shapes=[]
		plot_far.widget.add_vline(x = input.year(), line_width=3, line_dash="dot",annotation_text=str(input.year()), 
			annotation_position="bottom right",
			line_color="blue")
		plot_dr.widget.layout.annotations=[]
		plot_dr.widget.layout.shapes=[]
		plot_dr.widget.add_vline(x = input.year(), line_width=3, line_dash="dot",annotation_text=str(input.year()), 
			annotation_position="bottom right",
			line_color="blue")
		plot_rl.widget.layout.annotations=[]
		plot_rl.widget.layout.shapes=[]
		plot_rl.widget.add_vline(x = input.year(), line_width=3, line_dash="dot",annotation_text=str(input.year()), 
			annotation_position="bottom right",
			line_color="blue")
		
		 
	@reactive.calc
	@reactive.event(input.go_dr, ignore_none=True, ignore_init=True)
	def dr_calc():

		input.go_dr()

		with reactive.isolate():
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()
			p = input.dr()

		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		## Load models and observations
		##=============================
		Xo,Yo_tmp = load_obs( pathInp , lat_sub, lon_sub )

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
		climMM_file = os.path.join( pathInp , "climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', **bayes_kwargs)

		## Stats
		upper_side = "upper"
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC

		rl_cons = xr.DataArray( np.zeros( (ny,nsample_MCMC) ) , coords = [climCXCB.X.time , samples_MCMC] , dims = ["time","sample"] )
		law_cons = climCXCB.ns_law

		for s_MCMC in samples_MCMC:
			
			s_X = str(s_MCMC.values).split("_")[0]

			## Start with params
			law_cons.set_params( climCXCB.law_coef.loc[:,s_MCMC,'Multi_Synthesis'].values )

			## Go to factual world
			law_cons.set_covariable( climCXCB.X.loc[:,s_X,"F",'Multi_Synthesis'].values , climCXCB.time )

			## Find value of event definition
			rl_cons.loc[:,s_MCMC] = np.repeat(bias["Multi_Synthesis"].values,ny) + law_cons.isf( 1./p , climCXCB.time ) #if upper_side else law.icdf( pF , event.time ) )
		
		print("rp done")            
		return rl_cons

	@render_widget
	def plot_dr():

		txx_cons = pd.DataFrame({"year": dr_calc().time , "return_level": dr_calc().loc[:,'BE']})
		txx_cons = txx_cons.set_index("year")

		vInf = np.percentile(dr_calc(), qInf, axis=1)
		vSup = np.percentile(dr_calc(), qSup, axis=1)
		txx_cons["q%s" % int(qInf)] = vInf
		txx_cons["q%s" % int(qSup)] = vSup

		fig_dr = go.Figure()

		fig_dr.add_trace(go.Scatter(
			x=np.concatenate([txx_cons.index.values, txx_cons.index.values[::-1]]),
			y=np.concatenate([txx_cons["q95"], txx_cons["q5"][::-1]]),
			yhoverformat='.2f',
			fill='toself',
			hoveron='points',
			line=dict(color="blue"),
			showlegend=False,
			name="Confidence interval (95%)"
		))

		fig_dr.add_trace(go.Scatter(
			x=txx_cons.index.values,
			y=txx_cons["return_level"],
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="blue"),
			hoveron='points',
			name="Return level"
		))

		fig_dr.update_yaxes(title_text="Temperature (°C)")
		#fig_dr.update_layout(title="Return period of "+str(input.year())+" years at "+str(round(cons_calc()[2],2))+"°N / "+str(round(cons_calc()[3],2))+" °E")

		return fig_dr

	@reactive.calc
	@reactive.event(input.go_rl, ignore_none=True, ignore_init=True)
	def rl_calc():

		input.go_rl()

		with reactive.isolate():
			lat_sub = input.coord_lat()
			lon_sub = input.coord_lon()
			rl_user = input.rl_val()+273.15

		idx_lat = list(lat.lat.values).index(lat.sel(lat=lat_sub,method="nearest").lat)
		idx_lon = list(lon.lon.values).index(lon.sel(lon=lon_sub,method="nearest").lon)

		## Load models and observations
		##=============================
		Xo,Yo_tmp = load_obs( pathInp , lat_sub, lon_sub )

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
		climMM_file = os.path.join( pathInp , "climMM_lat%s_lon%s.nc" % (idx_lat,idx_lon) )
		climMM = ns.Climatology.from_netcdf( climMM_file, ns_law )

		## Apply constraints
		##==================
		climCX     = ns.constrain_covariate( climMM , Xo , time_reference , assume_good_scale = True, verbose = verbose )

		bayes_kwargs = { "n_ess"   : int(10000/(len(climCX.data.sample)-1))   } #Ici produit 10000 tirages en tout
		climCXCB = ns.stan_constrain(climCX,Yo,'stan_files/GEV_non_stationary.stan', **bayes_kwargs)

		## Stats
		upper_side = "upper"
		time = climCXCB.time
		ny = climCXCB.n_time
		nsample = climCXCB.n_sample
		samples = climCXCB.sample
		nsample_MCMC = climCXCB.data.sample_MCMC.shape[0]
		samples_MCMC = climCXCB.data.sample_MCMC

		rl = rl_user-bias["Multi_Synthesis"].values

		rp_cons = xr.DataArray( np.zeros( (ny,nsample_MCMC) ) , coords = [climCXCB.X.time , samples_MCMC ] , dims = ["time","sample"] )
		law_cons = climCXCB.ns_law

		for s_MCMC in samples_MCMC:
			
			s_X = str(s_MCMC.values).split("_")[0]

			## Start with params
			law_cons.set_params( climCXCB.law_coef.loc[:,s_MCMC,'Multi_Synthesis'].values )

			## Go to factual world
			law_cons.set_covariable( climCXCB.X.loc[:,s_X,"F",'Multi_Synthesis'].values , climCXCB.time )

			## Find value of event definition
			rp_cons.loc[:,s_MCMC] = 1/law_cons.sf( rl , climCXCB.time ) #if upper_side else law.icdf( pF , event.time ) )

		print("rl done")    
		return rp_cons
	
	@render_widget
	def plot_rl():

		txx_cons = pd.DataFrame({"year": rl_calc().time , "return_period": rl_calc().loc[:,'BE']})
		txx_cons = txx_cons.set_index("year")

		# Sélection des quantiles
		vInf = np.percentile(rl_calc(), qInf, axis=1)
		vSup = np.percentile(rl_calc(), qSup, axis=1)
		txx_cons["q%s" % int(qInf)] = vInf
		txx_cons["q%s" % int(qSup)] = vSup

		fig_dr = go.Figure()

		fig_dr.add_trace(go.Scatter(
			x=np.concatenate([txx_cons.index.values, txx_cons.index.values[::-1]]),
			y=np.concatenate([txx_cons["q95"], txx_cons["q5"][::-1]]),
			yhoverformat='.2f',
			fill='toself',
			hoveron='points',
			line=dict(color="blue"),
			showlegend=False,
			name="Confidence interval (95%)"
		))

		fig_dr.add_trace(go.Scatter(
			x=txx_cons.index.values,
			y=txx_cons["return_period"],
			yhoverformat='.2f',
			mode='lines',
			line=dict(color="blue"),
			hoveron='points',
			name="Return period"
		))

		fig_dr.update_yaxes(title_text="Return period (years)")
		#fig_dr.update_layout(title="Return period of "+str(input.year())+" years at "+str(round(cons_calc()[2],2))+"°N / "+str(round(cons_calc()[3],2))+" °E")

		return fig_dr


app = App(app_ui, server)
