def reformat_DRM(drm):
    """
    PROCESS THE EXOSIMS RESULTS DRM OBJECT IN A MORE PALLATABLE FORMAT.
    DRM IS CURRENTLY A LIST OF DICTIONARY THAT CONTAINS INFORMATION FOR EVERY VISIT.
    THIS REFORMATTER TURNS IT INTO A DICTIONARY OF RESULTS

    """
    # ========================================================================
    # WHAT KEYS WILL BE ORGANIZED HOW - UPDATED AS NEEDED
    # ------------------------------------------------------------------------
    # Don't do anything with these initially- these are dictionaries in the DRM.
    dontdoanything =['det_params','char_params']

    # Keys whose array elements need to be repeated based on number of planets
    # detected per star.
    expand_keys = ['star_name','star_ind','arrival_time','OB_nb',
                   'det_mode','char_mode', 'det_time','char_time',
                   'det_fZ', 'char_fZ',
                   'FA_det_status','FA_char_status','FA_char_SNR','FA_char_fEZ',
                   'FA_char_dMag','FA_char_WA',
                   'slew_time','slew_dV','slew_mass_used',
                   'sc_mass','det_dV','det_mass_used',
                   'det_dF_lateral','det_dF_axial','char_dV','char_mass_used',
                   'char_dF_lateral','char_dF_axial']


    # Keys whose arrays need to be concatendated because each is an array
    # e.g. - plan_inds: [ [0],[0,21,5],...,[51,20] ] <-- where each sub-array
    # corresponds to n planets around one star. 
    # In the end, each element in these arrays will have their
    # own individually mapped index in arrays.

    concatenate_keys = ['plan_inds',
                        'det_status','char_status',
                        'det_SNR', 'char_SNR']

    # Keys whose arrays will be expanded from 'char_params', and 'det_params'
    # UPDATE AS NEDED.
    det_expandkeys = ['WA','d','dMag','fEZ','phi']
    char_expandkeys = ['WA','d','dMag','fEZ','phi']
    # ========================================================================
    
    # CHECK IF DRM IS EMPTY
    if drm:
        
        # ====================================================================
        # COLLECT ALL KEYS IN DRM AND UNIQUE-IFY THE ARRAY
        kys = np.array([dt.keys() for dt in drm])
        kys = np.unique(np.concatenate(kys))
        # ====================================================================
        
        # --------------------------------------------
        # create dictionary of drm based on keywords
        ddrm = {}
        # --------------------------------------------
        
        # ====================================================================
        #        REARRANGE THE DRM
        # ----------------------------------------------------
        # change DRM from array of diciontaries to dictionary 
        # where the values of each key is an array. Each entry 
        # in the arrays represents the meta data for a single planet.
        # This section will do the above, and replace bad value stuff
        # with nan's. Keys reported in DRM not in the ICD are not used.
        for ky in kys:
            arr = []
            for dt in drm:

                if ky not in dt:
                    arr.append([np.nan])
                elif isinstance(dt[ky],np.ndarray):
                    if np.size(dt[ky]) == 0:
                        arr.append([np.nan])
                    else: arr.append(dt[ky])
                elif not isinstance(dt[ky],(int,long,float)) and not dt[ky]:
                    arr.append([np.nan])

                else:
                    arr.append(dt.get(ky,[np.nan]))

            ddrm[ky] = np.array(arr)
        # ====================================================================
        
        # ====================================================================
        #  EXPAND ARRAYS TO GET SINGLE PLANET MAPPING
        # --------------------------------------------------------------------
        plan_raw_inds = ddrm['plan_inds']

        # Here, clone the values for each "expand_keys" key to the number of
        # planets per star.
        for ky in expand_keys:
            try:
                vals = ddrm[ky]
                # ADD SAME # AS PLANETS DETECTED.
                # AT LEAST ONE ADDED EVEN WITH NO PLANETS DETECTED
                temp_val = [[vals[i]] * len(plan_raw_inds[i])
                            for i in xrange(len(plan_raw_inds))]

                ddrm[ky] = np.concatenate(temp_val)
            except KeyError:
                pass
        # ====================================================================

        # ====================================================================
        #  CONCATENATE ARRAYS TO GET SINGLE PLANET MAPPING
        # --------------------------------------------------------------------        
        for ky in concatenate_keys:
            try:
                ddrm[ky] = np.concatenate(ddrm[ky])
            except KeyError:
                pass
        # ====================================================================
        
        # ====================================================================
        # UNZIP THE VALUES IN 'det_params" and "char_params" to a one-to-one 
        # mapping with each planet.
        # remove_astropy strips each value of its astropy quantity... cause why the 
        # hell is that in the output?!!
        
        if ddrm.has_key('det_params'):
            det_DRM = list(ddrm['det_params'])
            det_DRM = remove_astropy(det_DRM)
            
            for ky in det_expandkeys:
                arr = np.array([dt['WA'] for dt in det_DRM])
                arr = np.concatenate(arr)
                ddrm['det_'+ky] = arr
            # REMOVE Zipped dicitonary
            ddrm.pop('det_params');
            
        if ddrm.has_key('char_params'):
            char_DRM = list(ddrm['char_params'])
            char_DRM = remove_astropy(char_DRM)
            
            for ky in char_expandkeys:
                arr = np.array([dt['WA'] for dt in char_DRM])
                arr = np.concatenate(arr)
                ddrm['char_'+ky] = arr
            # REMOVE Zipped dicitonary
            ddrm.pop('char_params');
                
    else:
        ddrm = None

    return ddrm


def remove_astropy(DRM):
    
    """
    Given the DRM from an EXOSIMS simulation, strip the top level 
    astropy quantity to bare numbers, because why the hell is an 
    astropy quant IN THE DAMN OUTPUT?!!
    """
    
    kys = np.array([dt.keys() for dt in DRM])
    kys = np.unique(np.concatenate(kys))
    
    if DRM: 
        for i in xrange(len(DRM)):

            for ky in kys:
                if isinstance(DRM[i][ky], astropy.units.quantity.Quantity):
                    DRM[i][ky] = DRM[i][ky].value
                
    return DRM
            
