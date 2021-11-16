from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from astropy.io import fits

from .transformations import polar_to_cart, cart_to_polar, rotate
from .utilities import query_simbad, convert_spt_to_pandeia, user_spectrum, pandeia_spectrum, normalise_spectrum, compute_magnitude
from pandeia.engine.calc_utils import build_default_calc


class Scene():
    '''
    Scene class to mirror the construction of a typical Pandeia 'Scene', reinventing the wheel a little bit, but means that
    users don't need to import pandeia in their main code and that aspects of data input can be streamlined. 
    '''
    __NEXT_ID = 1
    def __init__(self, name=None, **kwargs):
        '''
        Parameters
        ----------
        name : str
            Name of the scene
        '''
        #Load a default pandeia scene to assign properties to. NIRCam/Coroangraphy doesn't matter here, just need an empty scene dict

        self.pandeia_scene = build_default_calc('jwst', 'nircam', 'coronagraphy')['scene']
        self.pandeia_scene[0]['assigned'] = False #No source has been assigned to this 'default' scene yet 
        self.source_list = []

        if name == None:
            self.scene_name = 'Scene{}'.format(Scene.__NEXT_ID)
            Scene.__NEXT_ID += 1
        else:
            self.scene_name = name

    def add_source(self, name, kind='simbad', r=0.0, theta=0.0, verbose=True, **kwargs):
        '''
        Add a source to the Scene

        Parameters
        ----------
        name : str
            Name of the source to be added
        kind : str
            Kind of source to add, options are
            - 'simbad' for SIMBAD query of name string
            - 'grid' to use Pandeia Phoenix  with 'spt', 'norm_val', 'norm_unit', 'norm_bandpass' passed as kwargs
            - 'file' to use input file, with 'wave_unit' and 'flux_unit' passed as kwargs
        r : float
            Radial separation of source from center in arcseconds
        theta : float
            PA of source, should be N->E counterclockwise
        verbose : bool
            Print update statements to terminal
        '''
        if verbose: print('{} // Adding Source: {}'.format(self.scene_name, name))
        raw_id = len(self.source_list)
        
        if '_' in name or ':' in name:
            raise ValueError('Underscores "_" and colons ":" cannot be included in Scene names for file saving purposes')

        self.source_list.append(name) 
        #Check if the scene dictionary needs to be extended to add another source. 
        if raw_id > 0: self.pandeia_scene.append(deepcopy(self.pandeia_scene[0]))

        #Load in source
        working_source = self.pandeia_scene[raw_id]
        working_source['id'] = raw_id+1 #Each source must have an allocated source ID starting at 1. 
        working_source['pancake_parameters'] = {}
        working_source['pancake_parameters']['name'] = name
        
        #Apply source offset
        #NOTE: polar_to_cart returns a normal conversion, in our case things are flipped as we are working from the y-axis. 
        yoff, xoff = polar_to_cart(r, theta)
        working_source['position']['x_offset'] = -xoff #Negative as we are working N->E counterclockwise
        working_source['position']['y_offset'] = yoff

        # Use 'kind' of source input to read in different properties 
        if kind == 'simbad':
            #Attempt to query data for the input 'name' string from simbad
            query_results = query_simbad(name, verbose=verbose)

            approx_spt = convert_spt_to_pandeia(query_results['spt'])
            working_source['pancake_parameters']['spt'] = approx_spt
            for qresult in ['ra', 'dec', 'norm_bandpass', 'norm_val', 'norm_unit']:
                working_source['pancake_parameters'][qresult] = query_results[qresult]

            #Generate spectrum
            raw_spectrum_wave, raw_spectrum_flux = pandeia_spectrum(working_source['pancake_parameters']['spt'])
            spectrum_wave, spectrum_flux = normalise_spectrum(raw_spectrum_wave, raw_spectrum_flux, norm_val=working_source['pancake_parameters']['norm_val'], norm_unit=working_source['pancake_parameters']['norm_unit'], norm_bandpass=working_source['pancake_parameters']['norm_bandpass'])
        elif kind == 'grid':
            #User provides the spectral type, normalisation bandpass, and normalisation mag for source so a spectrum can be retrieved from a grid
            for ginput in ['spt', 'norm_val', 'norm_unit', 'norm_bandpass']:
                #Check variable has been provided
                try: 
                    ginput_val = kwargs.get(ginput)
                except:
                    raise NameError("Please provide the spectral type ('spt'), normalisation flux ('norm_val'), normalisation flux unit ('norm_unit'), and normalisation bandpass ('norm_bandpass') of the source")

                #Check inputs are of the correct variable types.
                if ginput!='norm_val' and not isinstance(ginput_val, str):
                    raise TypeError("{} input must be of string type.".format(ginput))
                elif ginput=='norm_val' and not isinstance(ginput_val, (int,float)):
                    raise TypeError("{} input must be of int or float type.".format(ginput))
                
                # Assign input values to the scene. 
                if ginput == 'spt':
                    #Find the best approximation spectral type that Pandeia understands (could match what user provides).
                    approx_spt = convert_spt_to_pandeia(kwargs.get(ginput))
                    working_source['pancake_parameters']['spt'] = approx_spt
                else:
                    working_source['pancake_parameters'][ginput] = kwargs.get(ginput)

            #Generate spectrum
            raw_spectrum_wave, raw_spectrum_flux = pandeia_spectrum(working_source['pancake_parameters']['spt'])
            spectrum_wave, spectrum_flux = normalise_spectrum(raw_spectrum_wave, raw_spectrum_flux, norm_val=working_source['pancake_parameters']['norm_val'], norm_unit=working_source['pancake_parameters']['norm_unit'], norm_bandpass=working_source['pancake_parameters']['norm_bandpass'])
        elif kind == 'file':
            #User provides a file location for a spectrum which can then be read in and converted to microns and mJy
            spectrum_wave, spectrum_flux = user_spectrum(kwargs.get('filename'), wave_unit=kwargs.get('wave_unit'), flux_unit=kwargs.get('flux_unit'))
        else:
            raise ValueError("Source generation kind not recognised, please use 'simbad', 'grid', or 'file'")

        #We are taking the normalisation away from Pandeia so that other filters can be added like 2MASS/WISE.
        working_source['spectrum']['normalization']['type'] = 'none' 
        #Assign spectrum of source to the Pandeia scene.  
        working_source['spectrum']['sed']['sed_type'] = 'input'
        working_source['spectrum']['sed']['spectrum'] = [spectrum_wave, spectrum_flux]

    def renormalize_source(self, *args, **kwargs):
        '''
        I assure you it's pronounced 'zed'.
        '''
        return renormalise_source(self, *args, **kwargs)

    def renormalise_source(self, source, norm_val=5, norm_unit='vegamag', norm_bandpass='2mass_ks'):
        ''' 
        Renormalise a source already within a scene.

        Parameters
        ----------
        source : str
            Name of source to renormalise
        norm_val : float
            Value to renormalise to
        norm_unit : str
            Unit to perform renormalisation to
        norm_bandpass : str
            String for the bandpass we are normalising under, 2MASS, WISE or anything in synphot by default.  
        '''
        try:
            raw_id = self.source_list.index(source)
        except:
            raise ValueError('Source {} has not been allocated to this scene. Currently allocated sources are: {}'.format(source, ', '.join(self.source_list)))

        working_source = self.pandeia_scene[raw_id]

        spectrum_wave, spectrum_flux = working_source['spectrum']['sed']['spectrum']
        renorm_spec_wave, renorm_spec_flux = normalise_spectrum(spectrum_wave, spectrum_flux, norm_val=norm_val, norm_unit=norm_unit, norm_bandpass=norm_bandpass)

        working_source['spectrum']['sed']['spectrum'] = [renorm_spec_wave, renorm_spec_flux]

    def source_magnitude(self, source, filt):
        '''
        Calculate the magnitude of particular source in a given filter

        Parameters
        ----------
        source : str
            Name of source
        filt : str
            Filter to calculate magnitude in

        Returns
        -------
        magnitude : float
            Magnitude of object in apparent vegamag
        '''
        try:
            raw_id = self.source_list.index(source)
        except:
            raise ValueError('Source {} has not been allocated to this scene. Currently allocated sources are: {}'.format(source, ', '.join(self.source_list)))

        spectrum_wave, spectrum_flux = source['spectrum']['sed']['spectrum']

        magnitude = compute_magnitude(spectrum_wave, spectrum_flux, bandpass)

        return magnitude

    def offset_scene(self,x,y):
        '''
        Offset scene in x, y space in arcseconds

        Parameters
        ----------
        x : float
            x offset in arcseconds
        y : float
            y offset in arcseconds
        '''
        for source in self.pandeia_scene:
            source['position']['x_offset'] += x
            source['position']['y_offset'] += y

    def rotate_scene(self, theta, center=[0.,0.], direction='counter_clockwise'):
        '''
        Rotate scene given an angle in degrees
        
        Parameters
        ----------
        theta : float
            Angular distance to rotate
        center : list
            x and y coordinate to act as rotation center
        direction : str
            Direction to perform rotation, 'clockwise' or 'counter_clockwise'
            default is counter clockwise. 
        '''
        if direction == 'counter_clockwise': 
            #Subtract from 360 to convert to a counter clockwise rotation
            theta = 360-theta
        elif direction != 'clockwise':
            raise ValueError('Invalid direction: {}, options are "clockwise" or "counter_clockwise"')
        for source in self.pandeia_scene:
            newxy = rotate([source['position']['x_offset'],source['position']['y_offset']],theta,center)
            source['position']['x_offset'] = newxy[0]
            source['position']['y_offset'] = newxy[1]
    
    def plot_source_spectra(self, sources='all', title='', newfig=True):
        '''
        Produce a plot of the spectra of sources within a scene. 

        Parameters
        ----------
        sources : str / list of strings
            List of source names, or 'all' to plot all source spectra
        title : str
            Title of plot
        newfig : bool
            Start a new figure, default is True
        '''
        if newfig:
            plt.figure(figsize=(8,5))
            ax = plt.subplot(111)
        for s in self.pandeia_scene:
            if s['pancake_parameters']['name'] in sources or sources == 'all':
                ax.plot(s['spectrum']['sed']['spectrum'][0], s['spectrum']['sed']['spectrum'][1], label=s['pancake_parameters']['name'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5,30)
        ax.set_ylim(1e-3,None)
        ax.set_title(title,y=1.1,fontsize=14)
        ax.tick_params(which='both', direction='in', labelsize=12, axis='both', top=True, right=True)
        ax.xaxis.set_ticklabels([], minor=True)
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_locator(tck.FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]))
        ax.xaxis.set_major_locator(tck.FixedLocator([0.6, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 28]))
        ax.set_xlabel('Wavelength ($\mu$m)',fontsize=14)
        ax.set_ylabel('Spectral Flux Density (mJy)',fontsize=14)
        ax.legend(numpoints=1,loc='best')
        plt.show()

    def plot_scene(self, title='', newfig=True):
        '''
        Plot the scene and its sources spatially. 

        Parameters
        ----------
        title : str
            Title of plot
        newfig : bool
            Start a new figure, default is True
        '''

        if newfig:
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111,projection='polar')
        for s in self.pandeia_scene:
            r, theta = cart_to_polar([s['position']['x_offset'],s['position']['y_offset']])
            theta -= 90 # As we use the y-axis as theta=0, not x
            ax.plot(np.deg2rad(theta),r,lw=0,marker='o',ms=10,label=s['pancake_parameters']['name'])
        ax.set_rmin(0) #Centre the scene at 0,0
        ax.set_title(title,y=1.1,fontsize=14)
        ax.legend(numpoints=1, loc='best', framealpha=1)
        ax.set_theta_offset(np.pi/2)
        plt.show()


def create_SGD(ta_error='none', fsm_error='default', stepsize=20.e-3, pattern_name=None, sim_num=0):
    '''
    Create small grid dither pointing set. There are two
    ways to specify dither patterns:

    ta_error - add TA error to each point in the SGD?

    stepsize - floating point value for a 3x3 grid.

    pattern_name - string name of a pattern corresponding to
              one of the named dither patterns in APT.

    If you specify pattern_name, then stepsize is ignored.

    See https://jwst-docs-stage.stsci.edu/display/JTI/NIRCam+Small-Grid+Dithers
    for information on the available dither patterns and their names.
    
    Parameters
    ----------
    ta_error : str / int / float
        Target acquisition error
        - 'saved' to use a saved random seed of offsets
        - 'random' for a random error
        - int or float for random error with set amplitude in mas
        - 'none' for no error
    fsm_error : str 
        Options of 'none' for no error, or 'default' for 2e-3 mas
    stepsize : float
        Manual step size for dither
    pattern_name : str
        Default small grid dither pattern name
    sim_num : int
        Specific simulation number / draw from a 'saved' ta_error

    Returns
    -------
    sgds : list
        list of x, y offsets for each step in the dither pattern. 

    '''
    
    # Small grid dither patterns
    if pattern_name is not None:
        pattern_name = pattern_name.upper()
        if pattern_name == "5-POINT-BOX":
            pointings = [(0,       0),
                         (0.015,   0.015),
                         (-0.015,  0.015),
                         (-0.015, -0.015),
                         (0.015,  -0.015)]
        elif pattern_name == "5-POINT-DIAMOND":
            pointings = [(0,      0),
                         (0,      0.02),
                         (0,     -0.02),
                         (+0.02,  0),
                         (-0.02,  0)]
        elif pattern_name == '9-POINT-CIRCLE':
            pointings = [( 0,      0),
                         ( 0,      0.02),
                         (-0.015,  0.015),
                         (-0.02,   0),
                         (-0.015, -0.015),
                         ( 0.000, -0.02),
                         ( 0.015, -0.015),
                         ( 0.020,  0.0),
                         ( 0.015,  0.015)]
        elif pattern_name == "3-POINT-BAR":
            pointings = [(0,    0),
                         (0.0,  0.015),
                         (0.0, -0.015)]
        elif pattern_name == "5-POINT-BAR":
            pointings = [(0,    0),
                         (0.0,  0.020),
                         (0.0,  0.010),
                         (0.0, -0.010),
                         (0.0, -0.020)]
        elif pattern_name == "SINGLE-POINT":
            pointings = [(0, 0)]
        elif pattern_name == "5-POINT-SMALL-GRID":
            pointings = [( 0,      0),
                         (-0.010,  0.010),
                         ( 0.010,  0.010),
                         ( 0.010, -0.010),
                         (-0.010, -0.010)]
        elif pattern_name == "9-POINT-SMALL-GRID":
            pointings = [( 0,      0),
                         (-0.010,  0.0),
                         (-0.010,  0.010),
                         ( 0.0,    0.010),
                         ( 0.010,  0.010),
                         ( 0.010,  0.0),
                         ( 0.010, -0.010),
                         ( 0.0,   -0.010),
                         (-0.010, -0.010)]
        else:
            raise ValueError("Unknown pattern_name value; check your input matches exactly an allowed SGD pattern in APT.")
    else:
        steps = [-stepsize,0.,stepsize]
        pointings = itertools.product(steps,steps)
    
    if ta_error=='saved':
        # Use a ta_error from a "saved" list of draws from a 5mas normal distribution (still randomnly generated, but seed is fixed)
        rngx = np.random.RandomState(42)
        rngy = np.random.RandomState(2021)
        saved_ta_x = rngx.normal(loc=0.,scale=5e-3,size=50)
        saved_ta_y = rngy.normal(loc=0.,scale=5e-3,size=50)
        ta_x, ta_y = saved_ta_x[sim_num], saved_ta_y[sim_num]
    elif ta_error=='random':
        # Simulate the TA error from a 5mas normal distribution
        ta_x, ta_y = get_ta_error(error='default')
    elif isinstance(ta_error, (int, float)):
        # Simulate the TA error from an X normal distribution (should provide in arcsec)
        ta_x, ta_y = get_ta_error(error=ta_error)
    elif ta_error=='none':
        ta_x, ta_y = 0., 0.
        fsm_error = 'none'
    else:
        raise ValueError('Target Acquisition (ta_error) string not recognised, options are "random", "saved", or "none", or user specified values.')
    
    sgds = []
    for i, (sx, sy) in enumerate(pointings):
        if i > 0:
            errx, erry = get_fsm_error(error=fsm_error)
            offset_x = sx + errx + ta_x
            offset_y = sy + erry + ta_y
        else:
            offset_x = sx + ta_x
            offset_y = sy + ta_y
        sgds.append([offset_x, offset_y])
    return sgds

def get_ta_error(error='default'):
    ''' 
    5mas 1-sigma/axis error (~7mas radial)

    Parameters
    ----------
    error : str / float
        String of 'default' for 5e-3, or input float

    Returns
    -------
    x,y random error
    '''
    if error == 'default': 
        error = 5e-3
    return np.random.normal(loc=0.,scale=error,size=2)

def get_fsm_error(error='default'):
    '''2mas 1/sigma/axis error from the fine steering mirror

    Parameters
    ----------
    error : str / float
        String of 'default' for 2e-3, 'none' for 0, or input float

    Returns
    -------
    x,y random error
    '''
    if error == 'default':
        error = 2e-3
    elif error == 'none':
        error = 0.

    return np.random.normal(loc=0.,scale=error,size=2)

