def reformat_DRM(drm):
    """
    PROCESS THE EXOSIMS RESULTS DRM OBJECT IN A MORE PALLATABLE FORMAT.
    DRM IS CURRENTLY A LIST OF DICTIONARY THAT CONTAINS INFORMATION FOR EVERY VISIT.
    THIS REFORMATTER TURNS IT INTO A DICTIONARY OF RESULTS

    """

    # list of keys to not do anything with -- in terms of expanding or concatenating.
    # THESE CAN BE UDPATED LATER
    dontdoanything = ['char_mode', 'slew_time',
                      'slew_angle', 'slew_dv',
                      'slew_mass_used', 'FA_fEZ',
                      'det_DV', 'det_mass_used',
                      'det_dF_lateral', 'det_dF_axial',
                      'char_dV', 'char_mass_used',
                      'char_dF_lateral', 'char_dF_axial', 'sc_mass']

    # list of keys whose array elements need to be repeated based on number of planets
    # detected per star.
    expand_keys = ['arrival_time', 'star_ind', 'det_time', 'char_time']

    # concatenate the arrays of these keys. In the end, each element in these arrays will have their
    # own individually mapped index in arrays in fill_keys.
    concatenate_keys = ['plan_inds', 'det_status',
                        'det_SNR', 'det_fEZ',
                        'det_dMag', 'det_WA',
                        'char_status', 'char_SNR',
                        'char_fEZ', 'char_dMag',
                        'char_WA', 'FA_status',
                        'FA_SNR', 'FA_dMag', 'FA_WA']

    # CHECK IF DRM IS EMPTY
    if drm:
        # COLLECT ALL KEYS IN DRM AND UNIQUE-IFY THE ARRAY
        kys = np.array([dt.keys() for dt in drm])
        kys = np.unique(np.concatenate(kys))

        # CREATE DICTIONARY OF DRM BASED ON KEYWORDS
        ddrm = {}

        for ky in kys:
            # ADD VALUE, UNLESS EMPTY OR KEY DOE

            
#            ddrm[ky] = np.array([[np.nan] if ky not in dt or
#                                          (not dt[ky].size and not isinstance(dt[ky], (int, long, float)))
#                                          else dt.get(ky, [np.nan]) for dt in drm])
            try:
                vals_list = []

                for dt in drm:

                    # CHECK IF EMPTY ARRAY
                    try: 
                        empty_bool = not dt[ky].size
                    except AttributeError:
                        empty_bool = not dt[ky]

                    # CHECK IF KY IS IN THE ith Dictionary OR is (not em

                    if ky not in dt or (empty_bool and not isinstance(dt[ky], (int,long, float))):

                        vals_list.append([np.nan])
                    else:
                        vals_list.append(dt.get(ky,[np.nan]))
                ddrm[ky] = np.array(vals_list)
            
            except ValueError: 
                ddrm[ky] = np.array([vals.value for vals in vals_list])

        plan_raw_inds = ddrm['plan_inds']

        # FILL OUT PROPER ARRAYS
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

        for ky in concatenate_keys:
            try:
                ddrm[ky] = np.concatenate(ddrm[ky])
            except KeyError:
                pass
    else:
        ddrm = None

    return ddrm

