from __future__ import print_function
import subprocess
import glob
import numpy as np
import healpy as hp
import os
import sys
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline


def zobov_wrapper(sample, use_vozisol=False, zobov_box_div=2, zobov_buffer=0.1):
    """Wrapper function to call C-based ZOBOV codes

    Arguments:
        sample: object of type Sample
        use_vozisol: flag to use vozisol.c to do tessellation (good for surveys)
        zobov_box_div: integer number of divisions of box, ignored if use_vozisol is True
        zobov_buffer: fraction of box length used as buffer region, ignored is use_vozisol is True

    """

    # ---run the tessellation--- #
    if use_vozisol:
        print("Calling vozisol to do the tessellation...")
        logfile = "./log/" + sample.handle + ".out"
        log = open(logfile, "w")
        cmd = ["./bin/vozisol", sample.posn_file, sample.handle, str(sample.box_length),
               str(sample.num_tracers), str(0.9e30)]
        subprocess.call(cmd, stdout=log, stderr=log)
        log.close()
    else:
        print("Calling vozinit, voz1b1 and voztie to do the tessellation...")

        # ---Step 1: call vozinit to write the script used to call voz1b1 and voztie--- #
        logfile = "./log/" + sample.handle + ".out"
        log = open(logfile, "w")
        cmd = ["./bin/vozinit", sample.posn_file, str(zobov_buffer), str(sample.box_length),
               str(zobov_box_div), sample.handle]
        subprocess.call(cmd, stdout=log, stderr=log)
        log.close()

        # ---Step 2: call this script to do the tessellation--- #
        voz_script = "scr" + sample.handle
        cmd = ["./%s" % voz_script]
        log = open(logfile, 'a')
        subprocess.call(cmd, stdout=log, stderr=log)
        log.close()

        # ---Step 3: check the tessellation was successful--- #
        if not os.access("%s.vol" % sample.handle, os.F_OK):
            sys.exit("Something went wrong with the tessellation. Aborting ...")

        # ---Step 4: remove the script file--- #
        if os.access(voz_script, os.F_OK):
            os.unlink(voz_script)

        # ---Step 5: copy the .vol files to .trvol--- #
        cmd = ["cp", "%s.vol" % sample.handle, "%s.trvol" % sample.handle]
        subprocess.call(cmd)

        # ---Step 6: if buffer mocks were used, remove them and flag edge galaxies--- #
        # (necessary because voz1b1 and voztie do not do this automatically)
        if sample.num_mocks > 0:
            cmd = ["./bin/checkedges", sample.handle, str(sample.num_tracers), str(0.9e30)]
            log = open(logfile, 'a')
            subprocess.call(cmd, stdout=log, stderr=log)
            log.close()

    print("Tessellation done.\n")

    # ---prepare files for running jozov--- #
    if sample.is_box:
        # no preparation is required for void-finding (no buffer mocks, no z-weights, no angular-weights)
        if sample.find_clusters:
            cmd = ["cp", "%s.vol" % sample.handle, "%sc.vol" % sample.handle]
            subprocess.call(cmd)
    else:
        # ---Step 1: read the edge-modified Voronoi volumes--- #
        with open('./%s.vol' % sample.handle, 'r') as F:
            npreal = np.fromfile(F, dtype=np.int32, count=1)
            modvols = np.fromfile(F, dtype=np.float64, count=npreal)

        # ---Step 2: renormalize volumes in units of mean volume per galaxy--- #
        # (this step is necessary because otherwise the buffer mocks affect the calculation)
        edgemask = modvols == 1.0/0.9e30
        modvols[np.logical_not(edgemask)] *= (sample.tracer_dens * sample.box_length ** 3.) / sample.num_part_total

        # ---Step 3: scale volumes accounting for z-dependent selection--- #
        if sample.use_z_wts:
            redshifts = sample.tracers[:sample.num_tracers, 5]
            selfnbins = np.loadtxt(sample.selection_fn_file)
            selfn = InterpolatedUnivariateSpline(selfnbins[:, 0], selfnbins[:, 2], k=1)
            # smooth with a Savitzky-Golay filter to remove high-frequency noise
            x = np.linspace(redshifts.min(), redshifts.max(), 1000)
            y = savgol_filter(selfn(x), 101, 3)
            # then linearly interpolate the filtered interpolation
            selfn = InterpolatedUnivariateSpline(x, y, k=1)
            # scale the densities according to this
            modfactors = selfn(redshifts[np.logical_not(edgemask)])
            modvols[np.logical_not(edgemask)] *= modfactors

        # ---Step 4: scale volumes accounting for angular completeness--- #
        if sample.use_ang_wts:
            ra = sample.tracers[:sample.num_tracers, 3]
            dec = sample.tracers[:sample.num_tracers, 4]
            # fetch the survey mask
            mask = hp.read_map(sample.mask_file, verbose=False)
            nside = hp.get_nside(mask)
            # weight the densities by completeness
            pixels = hp.ang2pix(nside, np.deg2rad(90 - dec), np.deg2rad(ra))
            modfactors = mask[pixels]
            modvols[np.logical_not(edgemask)] *= modfactors[np.logical_not(edgemask)]

        # ---Step 5: write the scaled volumes to file--- #
        with open("./%s.vol" % sample.handle, 'w') as F:
            npreal.tofile(F, format="%d")
            modvols.tofile(F, format="%f")

        # ---Step 6: if finding clusters, create the files required--- #
        if sample.find_clusters:
            modvols[edgemask] = 0.9e30
            # and write to c.vol file
            with open("./%sc.vol" % sample.handle, 'w') as F:
                npreal.tofile(F, format="%d")
                modvols.tofile(F, format="%f")

        # ---Step 7: set the number of non-edge galaxies--- #
        sample.num_non_edge = sample.num_tracers - sum(edgemask)

    # ---run jozov to perform the void-finding--- #
    cmd = ["./bin/jozovtrvol", "v", sample.handle, str(0), str(0)]
    log = open(logfile, 'a')
    subprocess.call(cmd)
    log.close()
    # this call to (modified version of) jozov sets NO density threshold, so
    # ALL voids are merged without limit and the FULL merged void heirarchy is
    # output to file; distinct voids are later obtained in post-processing

    # ---if finding clusters, run jozov again--- #
    if sample.find_clusters:
        cmd = ["./bin/jozovtrvol", "c", sample.handle, str(0), str(0)]
        log = open(logfile, 'a')
        subprocess.call(cmd)
        log.close()

    # ---clean up: remove unnecessary files--- #
    for fileName in glob.glob("./part." + sample.handle + ".*"):
        os.unlink(fileName)

    # ---clean up: move all other files to appropriate directory--- #
    raw_dir = sample.output_folder + "rawZOBOV/"
    if not os.access(raw_dir, os.F_OK):
        os.makedirs(raw_dir)
    for fileName in glob.glob("./" + sample.handle + "*"):
        cmd = ["mv", fileName, "%s." % raw_dir]
        subprocess.call(cmd)


def postprocess_voids(sample):
    """Method to post-process raw ZOBOV output to obtain discrete set of non-overlapping voids. This method
    is hard-coded to NOT allow any void merging, since no objective (non-arbitrary) criteria can be defined
    to control merging, if allowed.

    Arguments:
        sample: an object of class Sample
    """

    print('Post-processing voids ...\n')

    # ------------NOTE----------------- #
    # Actually, the current code is built from previous code that did have merging
    # functionality. This functionality is still technically present, but is controlled
    # by the following hard-coded parameters. If you know what you are doing, you can
    # change them.
    # --------------------------------- #
    dont_merge = True
    use_r_threshold = False
    r_threshold = 1.
    use_link_density_threshold = False
    link_density_threshold = 1.
    count_all_voids = True
    use_stripping = False
    strip_density_threshold = 1.
    if use_stripping:
        if (strip_density_threshold < sample.min_dens_cut) or (strip_density_threshold < link_density_threshold):
            print('ERROR: incorrect use of strip_density_threshold\nProceeding with automatically corrected value')
            strip_density_threshold = max(sample.min_dens_cut, link_density_threshold)
    # --------------------------------- #

    # the files with ZOBOV output
    zone_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.zone'
    void_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.void'
    list_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.txt'
    volumes_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.trvol'
    densities_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.vol'

    # new files after post-processing
    new_void_file = sample.output_folder + sample.void_prefix + ".void"
    new_list_file = sample.output_folder + sample.void_prefix + "_list.txt"

    # load the list of void candidates
    voidsread = np.loadtxt(list_file, skiprows=2)
    # sort in ascending order of minimum dens
    sorted_order = np.argsort(voidsread[:, 3])
    voidsread = voidsread[sorted_order]

    num_voids = len(voidsread[:, 0])
    vid = np.asarray(voidsread[:, 0], dtype=int)
    edgelist = np.asarray(voidsread[:, 1], dtype=int)
    vollist = voidsread[:, 4]
    numpartlist = np.asarray(voidsread[:, 5], dtype=int)
    rlist = voidsread[:, 9]

    # load the void hierarchy
    with open(void_file, 'r') as Fvoid:
        hierarchy = Fvoid.readlines()
    # sanity check
    nvoids = int(hierarchy[0])
    if nvoids != num_voids:
        sys.exit('Unequal void numbers in voidfile and listfile, %d and %d!' % (nvoids, num_voids))
    hierarchy = hierarchy[1:]

    # load the particle-zone info
    zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

    # load the VTFE volume information
    with open(volumes_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                     % (npart, sample.handle, sample.num_tracers))
        vols = np.fromfile(File, dtype=np.float64, count=npart)

    # load the VTFE density information
    with open(densities_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                     % (npart, sample.handle, sample.num_tracers))
        densities = np.fromfile(File, dtype=np.float64, count=npart)
        densities = 1. / densities

    # mean volume per particle in box (including all buffer mocks)
    meanvol_trc = (sample.box_length**3.)/sample.num_part_total

    # parse the list of structures, separating distinct voids and performing minimal pruning
    with open(new_void_file, 'w') as Fnewvoid:
        with open(new_list_file, 'w') as Fnewlist:

            # initialize variables
            counted_zones = np.empty(0, dtype=int)
            edge_flag = np.empty(0, dtype=int)
            wtd_avg_dens = np.empty(0, dtype=int)
            num__acc = 0

            for i in range(num_voids):
                coredens = voidsread[i, 3]
                voidline = hierarchy[sorted_order[i]].split()
                pos = 1
                num_zones_to_add = int(voidline[pos])
                finalpos = pos + num_zones_to_add + 1
                rval = float(voidline[pos + 1])
                rstopadd = rlist[i]
                num_adds = 0
                if rval >= 1 and coredens < sample.min_dens_cut and numpartlist[i] >= sample.void_min_num \
                        and (count_all_voids or vid[i] not in counted_zones):
                    # this void passes basic pruning
                    add_more = True
                    num__acc += 1
                    zonelist = vid[i]
                    total_vol = vollist[i]
                    total_num_parts = numpartlist[i]
                    zonestoadd = []
                    while num_zones_to_add > 0 and add_more:  # more zones can potentially be added
                        zonestoadd = np.asarray(voidline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                        dens = rval * coredens
                        rsublist = rlist[np.in1d(vid, zonestoadd)]
                        volsublist = vollist[np.in1d(vid, zonestoadd)]
                        partsublist = numpartlist[np.in1d(vid, zonestoadd)]
                        if dont_merge or (use_link_density_threshold and dens > link_density_threshold) or \
                                (use_r_threshold > 0 and max(rsublist) > r_threshold):
                            # cannot add these zones
                            rstopadd = rval
                            add_more = False
                            finalpos -= (num_zones_to_add + 1)
                        else:
                            # keep adding zones
                            zonelist = np.append(zonelist, zonestoadd)
                            num_adds += num_zones_to_add
                            total_vol += np.sum(volsublist)  #
                            total_num_parts += np.sum(partsublist)  #
                        pos += num_zones_to_add + 2
                        num_zones_to_add = int(voidline[pos])
                        rval = float(voidline[pos + 1])
                        if add_more:
                            finalpos = pos + num_zones_to_add + 1

                    counted_zones = np.append(counted_zones, zonelist)
                    if use_stripping:
                        member_ids = np.logical_and(densities[:] < strip_density_threshold,
                                                    np.in1d(zonedata, zonelist))
                    else:
                        member_ids = np.in1d(zonedata, zonelist)

                    # if using void "stripping" functionality, recalculate void volume and number of particles
                    if use_stripping:
                        total_vol = np.sum(vols[member_ids])
                        total_num_parts = len(vols[member_ids])

                    # check if the void is edge-contaminated (useful for observational surveys only)
                    if 1 in edgelist[np.in1d(vid, zonestoadd)]:
                        edge_flag = np.append(edge_flag, 1)
                    else:
                        edge_flag = np.append(edge_flag, 0)

                    # average density of member cells weighted by cell volumes
                    w_a_d = np.sum(vols[member_ids] * densities[member_ids]) / np.sum(vols[member_ids])
                    wtd_avg_dens = np.append(wtd_avg_dens, w_a_d)

                    # set the new line for the .void file
                    newvoidline = voidline[:finalpos]
                    if not add_more:
                        newvoidline.append(str(0))
                    newvoidline.append(str(rstopadd))
                    # write line to the output .void file
                    for j in range(len(newvoidline)):
                        Fnewvoid.write(newvoidline[j] + '\t')
                    Fnewvoid.write('\n')
                    if rstopadd > 10 ** 20:
                        rstopadd = -1  # only structures entirely surrounded by edge particles
                    # write line to the output _list.txt file
                    Fnewlist.write('%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n' % (vid[i], int(voidsread[i, 2]), coredens,
                                                                         int(voidsread[i, 5]), num_adds + 1,
                                                                         total_num_parts, total_vol * meanvol_trc,
                                                                         rstopadd))

    # tidy up the files
    # insert first line with number of voids to the new .void file
    with open(new_void_file, 'r+') as Fnewvoid:
        old = Fnewvoid.read()
        Fnewvoid.seek(0)
        topline = "%d\n" % num__acc
        Fnewvoid.write(topline + old)

    # insert header to the _list.txt file
    listdata = np.loadtxt(new_list_file)
    header = '%d non-edge tracers in %s, %d voids\n' % (sample.num_non_edge, sample.handle, num__acc)
    header = header + 'VoidID CoreParticle CoreDens Zone#Parts Void#Zones Void#Parts VoidVol(Mpc/h^3) VoidDensRatio'
    np.savetxt(new_list_file, listdata, fmt='%d %d %0.6f %d %d %d %0.6f %0.6f', header=header)

    # now find void centres and create the void catalogue files
    edge_flag = find_void_circumcentres(sample, num__acc, wtd_avg_dens, edge_flag)
    if sample.use_barycentres:
        if not os.access(sample.output_folder + "barycentres/", os.F_OK):
            os.makedirs(sample.output_folder + "barycentres/")
        find_void_barycentres(sample, num__acc, edge_flag, use_stripping, strip_density_threshold)


def find_void_circumcentres(sample, num_struct, wtd_avg_dens, edge_flag):
    """Method that checks a list of processed voids, finds the void minimum density centres and writes
    the void catalogue file.

    Arguments:
        sample: object of class Sample
        num_struct: integer number of voids after pruning
        wtd_avg_dens: float array of shape (num_struct,), weighted average void densities from post-processing
        edge_flag: integer array of shape (num_struct,), edge contamination flags
    """

    print("Identified %d voids. Now extracting circumcentres ..." % num_struct)

    # set the filenames
    densities_file = sample.output_folder + "rawZOBOV/" + sample.handle + ".vol"
    adjacency_file = sample.output_folder + "rawZOBOV/" + sample.handle + ".adj"
    list_file = sample.output_folder + sample.void_prefix + "_list.txt"
    info_file = sample.output_folder + sample.void_prefix + "_cat.txt"

    # load the VTFE density information
    with open(densities_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                     % (npart, sample.handle, sample.num_tracers))
        densities = np.fromfile(File, dtype=np.float64, count=npart)
        densities = 1. / densities

    # check whether tracer information is present, re-read in if required
    if not len(sample.tracers) == sample.num_part_total:
        sample.reread_tracer_info()
    # extract the x,y,z positions of the galaxies only (no buffer mocks)
    positions = sample.tracers[:sample.num_tracers, :3]

    list_array = np.loadtxt(list_file)
    v_id = np.asarray(list_array[:, 0], dtype=int)
    corepart = np.asarray(list_array[:, 1], dtype=int)

    # read and assign adjacencies from ZOBOV output
    with open(adjacency_file, 'r') as AdjFile:
        npfromadj = np.fromfile(AdjFile, dtype=np.int32, count=1)
        if not npfromadj == sample.num_tracers:
            sys.exit("npart = %d from adjacency file does not match num_tracers = %d!"
                     % (npfromadj, sample.num_tracers))
        partadjs = [[] for i in range(npfromadj)]  # list of lists to record adjacencies - is there a better way?
        partadjcount = np.zeros(npfromadj, dtype=np.int32)  # counter to monitor adjacencies
        nadj = np.fromfile(AdjFile, dtype=np.int32, count=npfromadj)  # number of adjacencies for each particle
        # load up the adjacencies from ZOBOV output
        for i in range(npfromadj):
            numtomatch = np.fromfile(AdjFile, dtype=np.int32, count=1)
            if numtomatch > 0:
                # particle numbers of adjacent particles
                adjpartnumbers = np.fromfile(AdjFile, dtype=np.int32, count=numtomatch)
                # keep track of how many adjacencies had already been assigned
                oldcount = partadjcount[i]
                newcount = oldcount + len(adjpartnumbers)
                partadjcount[i] = newcount
                # and now assign new adjacencies
                partadjs[i][oldcount:newcount] = adjpartnumbers
                # now also assign the reverse adjacencies
                # (ZOBOV records only (i adj j) or (j adj i), not both)
                for index in adjpartnumbers:
                    partadjs[index].append(i)
                partadjcount[adjpartnumbers] += 1

    if sample.is_box:
        info_output = np.zeros((num_struct, 9))
    else:
        info_output = np.zeros((num_struct, 11))
    circumcentre = np.empty(3)

    # loop over void cores, calculating circumcentres and writing to file
    for i in range(num_struct):
        # get adjacencies of the core particle
        coreadjs = partadjs[corepart[i]]
        adj_dens = densities[coreadjs]

        # get the 3 lowest density mutually adjacent neighbours of the core particle
        first_nbr = coreadjs[np.argmin(adj_dens)]
        mutualadjs = np.intersect1d(coreadjs, partadjs[first_nbr])
        if len(mutualadjs) == 0:
            circumcentre = np.asarray([0, 0, 0])
            edge_flag[i] = 2
        else:
            mutualadj_dens = densities[mutualadjs]
            second_nbr = mutualadjs[np.argmin(mutualadj_dens)]
            finaladjs = np.intersect1d(mutualadjs, partadjs[second_nbr])
            if len(finaladjs) == 0:  # something has gone wrong at tessellation stage!
                circumcentre = np.asarray([0, 0, 0])
                edge_flag[i] = 2
            else:  # can calculate circumcentre position
                finaladj_dens = densities[finaladjs]
                third_nbr = finaladjs[np.argmin(finaladj_dens)]

                # collect positions of the vertices
                vertex_pos = np.zeros((4, 3))
                vertex_pos[0, :] = positions[corepart[i], :]
                vertex_pos[1, :] = positions[first_nbr, :]
                vertex_pos[2, :] = positions[second_nbr, :]
                vertex_pos[3, :] = positions[third_nbr, :]
                if sample.is_box:  # need to adjust for periodic BC
                    shift_inds = abs(vertex_pos[0, 0] - vertex_pos[:, 0]) > sample.box_length / 2.0
                    vertex_pos[shift_inds, 0] += sample.box_length * np.sign(vertex_pos[0, 0] -
                                                                             vertex_pos[shift_inds, 0])
                    shift_inds = abs(vertex_pos[0, 1] - vertex_pos[:, 1]) > sample.box_length / 2.0
                    vertex_pos[shift_inds, 1] += sample.box_length * np.sign(vertex_pos[0, 1] -
                                                                             vertex_pos[shift_inds, 1])
                    shift_inds = abs(vertex_pos[0, 2] - vertex_pos[:, 2]) > sample.box_length / 2.0
                    vertex_pos[shift_inds, 2] += sample.box_length * np.sign(vertex_pos[0, 2] -
                                                                             vertex_pos[shift_inds, 2])

                # solve for the circumcentre; for more details on this method and its stability,
                # see http://www.ics.uci.edu/~eppstein/junkyard/circumcentre.html
                a = np.bmat([[2 * np.dot(vertex_pos, vertex_pos.T), np.ones((4, 1))],
                             [np.ones((1, 4)), np.zeros((1, 1))]])
                b = np.hstack((np.sum(vertex_pos * vertex_pos, axis=1), np.ones((1))))
                x = np.linalg.solve(a, b)
                bary_coords = x[:-1]
                circumcentre[:] = np.dot(bary_coords, vertex_pos)

        if sample.is_box:
            # put centre coords back within the fiducial box if they have leaked out
            if circumcentre[0] < 0 or circumcentre[0] > sample.box_length:
                circumcentre[0] -= sample.box_length * np.sign(circumcentre[0])
            if circumcentre[1] < 0 or circumcentre[1] > sample.box_length:
                circumcentre[1] -= sample.box_length * np.sign(circumcentre[1])
            if circumcentre[2] < 0 or circumcentre[2] > sample.box_length:
                circumcentre[2] -= sample.box_length * np.sign(circumcentre[2])

        # calculate void effective radius
        eff_rad = (3.0 * list_array[i, 6] / (4 * np.pi)) ** (1.0 / 3)

        # if required, write sky positions to file
        if sample.is_box:
            info_output[i] = [v_id[i], circumcentre[0], circumcentre[1], circumcentre[2], eff_rad,
                              (list_array[i, 2] - 1.), (wtd_avg_dens[i] - 1.), (wtd_avg_dens[i] - 1) * eff_rad ** 1.2,
                              list_array[i, 7]]
        else:
            # convert void centre position to observer coordinates
            centre_obs = circumcentre - sample.box_length / 2.0  # move back into observer coordinates
            rdist = np.linalg.norm(centre_obs)
            eff_angrad = np.degrees(eff_rad / rdist)
            # calculate the sky coordinates of the void centre
            # (this step also allows fallback check of undetected tessellation leakage)
            if (rdist >= sample.cosmo.get_comoving_distance(sample.z_min)) and (
                    rdist <= sample.cosmo.get_comoving_distance(sample.z_max)):
                centre_redshift = sample.cosmo.get_redshift(rdist)
                centre_dec = 90 - np.degrees(np.arccos(centre_obs[2] / rdist))
                centre_ra = np.degrees(np.arctan2(centre_obs[1], centre_obs[0]))
                if centre_ra < 0:
                    centre_ra += 360  # to get RA in the range 0 to 360
                mask = hp.read_map(sample.mask_file, verbose=False)
                nside = hp.get_nside(mask)
                pixel = hp.ang2pix(nside, np.deg2rad(90 - centre_dec), np.deg2rad(centre_ra))
                if mask[pixel] == 0:  # something has gone wrong at tessellation stage
                    centre_redshift = -1
                    centre_dec = -60
                    centre_ra = -60
                    eff_angrad = 0
                    edge_flag[i] = 2
            else:  # something has gone wrong at tessellation stage
                centre_redshift = -1
                centre_dec = -60
                centre_ra = -60
                eff_angrad = 0
                edge_flag[i] = 2
            info_output[i] = [v_id[i], centre_ra, centre_dec, centre_redshift, eff_rad, (list_array[i, 2] - 1.),
                              (wtd_avg_dens[i] - 1.), (wtd_avg_dens[i] - 1) * eff_rad ** 1.2, list_array[i, 7],
                              eff_angrad, edge_flag[i]]

    # save output data to file
    header = "%d voids from %s\n" % (num_struct, sample.handle)
    if sample.is_box:
        header = header + 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.3f %0.6f %0.6f %0.6f %0.6f', header=header)
    else:
        header = header + 'VoidID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v ' + \
                 'DensRatio Theta_eff(deg) EdgeFlag'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.3f %0.3f %0.4f %0.3f %0.6f %0.6f %0.6f %0.6f %d',
                   header=header)

    return edge_flag


def find_void_barycentres(sample, num_struct, edge_flag, use_stripping=False, strip_density_threshold=1.):
    """Method that checks a list of processed voids, finds the void barycentres and writes
    the void catalogue file.

    Arguments:
        sample: object of class Sample
        num_struct: integer number of voids after pruning
        edge_flag: integer array of shape (num_struct,), edge contamination flags
        use_stripping: bool,optional (default is False, don't change unless you know what you're doing!)
        strip_density_threshold: float, optional (default 1.0, not required unless use_stripping is True)
    """

    print('Now extracting void barycentres ...\n')

    # set the filenames
    vol_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.trvol'
    dens_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.vol'
    zone_file = sample.output_folder + 'rawZOBOV/' + sample.handle + '.zone'
    hierarchy_file = sample.output_folder + sample.void_prefix + '.void'
    list_file = sample.output_folder + sample.void_prefix + '_list.txt'
    info_file = sample.output_folder + 'barycentres/' + sample.void_prefix + '_baryC_cat.txt'

    # load up the particle-zone info
    zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

    # load the VTFE volume information
    with open(vol_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                     % (npart, sample.handle, sample.num_tracers))
        vols = np.fromfile(File, dtype=np.float64, count=npart)

    # load the VTFE density information
    with open(dens_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit("npart = %d in %s.vol file does not match num_tracers = %d!"
                     % (npart, sample.handle, sample.num_tracers))
        densities = np.fromfile(File, dtype=np.float64, count=npart)
        densities = 1. / densities

    # mean volume per particle in box (including all buffer mocks)
    meanvol_trc = (sample.box_length**3.)/sample.num_part_total

    # check whether tracer information is present, re-read in if required
    if not len(sample.tracers) == sample.num_part_total:
        sample.reread_tracer_info()
    # extract the x,y,z positions of the galaxies only (no buffer mocks)
    positions = sample.tracers[:sample.num_tracers, :3]

    list_array = np.loadtxt(list_file, skiprows=2)
    if sample.is_box:
        info_output = np.zeros((num_struct, 9))
    else:
        info_output = np.zeros((num_struct, 11))
    with open(hierarchy_file, 'r') as FHierarchy:
        FHierarchy.readline()  # skip the first line, contains total number of structures
        for i in range(num_struct):
            # get the member zones of the structure
            structline = (FHierarchy.readline()).split()
            pos = 1
            add_zones = int(structline[pos]) > 0
            member_zones = np.asarray(structline[0], dtype=int)
            while add_zones:
                num_zones_to_add = int(structline[pos])
                zonestoadd = np.asarray(structline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                member_zones = np.append(member_zones, zonestoadd)
                pos += num_zones_to_add + 2
                add_zones = int(structline[pos]) > 0

            # get the member particles for these zones
            if use_stripping:
                member_ids = np.logical_and(densities[:] < strip_density_threshold, np.in1d(zonedata, member_zones))
            else:  # stripDens functionality disabled
                member_ids = np.in1d(zonedata, member_zones)
            member_x = positions[member_ids, 0] - positions[int(list_array[i, 1]), 0]
            member_y = positions[member_ids, 1] - positions[int(list_array[i, 1]), 1]
            member_z = positions[member_ids, 2] - positions[int(list_array[i, 1]), 2]
            member_vols = vols[member_ids]
            member_dens = densities[member_ids]

            if sample.is_box:
                # must account for periodic boundary conditions, assume box coordinates in range [0,box_length]!
                shift_vec = np.zeros((len(member_x), 3))
                shift_x_ids = abs(member_x) > sample.box_length / 2.0
                shift_y_ids = abs(member_y) > sample.box_length / 2.0
                shift_z_ids = abs(member_z) > sample.box_length / 2.0
                shift_vec[shift_x_ids, 0] = -np.copysign(sample.box_length, member_x[shift_x_ids])
                shift_vec[shift_y_ids, 1] = -np.copysign(sample.box_length, member_y[shift_y_ids])
                shift_vec[shift_z_ids, 2] = -np.copysign(sample.box_length, member_z[shift_z_ids])
                member_x += shift_vec[:, 0]
                member_y += shift_vec[:, 1]
                member_z += shift_vec[:, 2]

            # volume-weighted barycentre of the structure
            centre = np.empty(3)
            centre[0] = np.sum(member_x * member_vols / np.sum(member_vols)) + positions[int(list_array[i, 1]), 0]
            centre[1] = np.sum(member_y * member_vols / np.sum(member_vols)) + positions[int(list_array[i, 1]), 1]
            centre[2] = np.sum(member_z * member_vols / np.sum(member_vols)) + positions[int(list_array[i, 1]), 2]

            # put centre coords back within the fiducial box if they have leaked out
            if sample.is_box:
                if centre[0] < 0 or centre[0] > sample.box_length:
                    centre[0] -= sample.box_length * np.sign(centre[0])
                if centre[1] < 0 or centre[1] > sample.box_length:
                    centre[1] -= sample.box_length * np.sign(centre[1])
                if centre[2] < 0 or centre[2] > sample.box_length:
                    centre[2] -= sample.box_length * np.sign(centre[2])

            # total volume of structure in Mpc/h, and effective radius
            void_vol = np.sum(member_vols) * meanvol_trc
            eff_rad = (3.0 * void_vol / (4 * np.pi)) ** (1.0 / 3)

            # average density of member cells weighted by cell volumes
            wtd_avg_dens = np.sum(member_dens * member_vols) / np.sum(member_vols)

            lambda_v = (wtd_avg_dens - 1) * eff_rad ** 1.2

            # if required, write sky positions to file
            if sample.is_box:
                info_output[i] = [list_array[i, 0], centre[0], centre[1], centre[2], eff_rad, (list_array[i, 2] - 1.),
                                  (wtd_avg_dens - 1.), lambda_v, list_array[i, 7]]
            else:
                centre_obs = centre - sample.box_length / 2.0  # move back into observer coordinates
                rdist = np.linalg.norm(centre_obs)
                eff_angrad = np.degrees(eff_rad / rdist)
                if (rdist >= sample.cosmo.get_comoving_distance(sample.z_min)) and (
                        rdist <= sample.cosmo.get_comoving_distance(sample.z_max)):
                    centre_redshift = sample.cosmo.get_redshift(rdist)
                    centre_dec = 90 - np.degrees(np.arccos(centre_obs[2] / rdist))
                    centre_ra = np.degrees(np.arctan2(centre_obs[1], centre_obs[0]))
                    if centre_ra < 0:
                        centre_ra += 360  # to get RA in the range 0 to 360
                    mask = hp.read_map(sample.mask_file, verbose=False)
                    nside = hp.get_nside(mask)
                    pixel = hp.ang2pix(nside, np.deg2rad(90 - centre_dec), np.deg2rad(centre_ra))
                    if mask[pixel] == 0:  # something has gone wrong at tessellation stage
                        centre_redshift = -1
                        centre_dec = -60
                        centre_ra = -60
                        eff_angrad = 0
                        edge_flag[i] = 2
                else:  # something has gone wrong at tessellation stage
                    centre_redshift = -1
                    centre_dec = -60
                    centre_ra = -60
                    eff_angrad = 0
                    edge_flag[i] = 2
                info_output[i] = [list_array[i, 0], centre_ra, centre_dec, centre_redshift, eff_rad,
                                  (list_array[i, 2] - 1.), (wtd_avg_dens - 1.), lambda_v, list_array[i, 7],
                                  eff_angrad, edge_flag[i]]

    # save output data to file
    header = "%d voids from %s\n" % (num_struct, sample.handle)
    if sample.is_box:
        header = header + 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.3f %0.6f %0.6f %0.6f %0.6f', header=header)
    else:
        header = header + 'VoidID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v' + \
                 'DensRatio Theta_eff(deg) EdgeFlag'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.3f %0.3f %0.4f %0.3f %0.6f %0.6f %0.6f %0.6f %d',
                   header=header)


def postprocess_clusters(sample):
    """Method to post-process raw ZOBOV output to obtain discrete set of non-overlapping 'superclusters'. This method
    is hard-coded to NOT allow any supercluster merging, since no objective (non-arbitrary) criteria can be defined
    to control merging anyway.

    Arguments:
        sample: an object of class Sample
    """

    print('Post-processing superclusters ...\n')

    # ------------NOTE----------------- #
    # Actually, the current code is built from previous code that did have merging
    # functionality. This functionality is still technically present, but is controlled
    # by the following hard-coded parameters. If you know what you are doing, you can
    # change them.
    # --------------------------------- #
    dont_merge = True
    use_r_threshold = False
    r_threshold = 2.
    use_link_density_threshold = False
    link_density_threshold = 1.
    count_all_clusters = True
    use_stripping = False
    strip_density_threshold = 1.
    if use_stripping:
        if (strip_density_threshold > sample.max_dens_cut) or (strip_density_threshold > link_density_threshold):
            print('ERROR: incorrect use of strip_density_threshold\nProceeding with automatically corrected value')
            strip_density_threshold = max(sample.max_dens_cut, link_density_threshold)
    # --------------------------------- #

    # the files with ZOBOV output
    zone_file = sample.output_folder + "rawZOBOV/" + sample.handle + "c.zone"
    clust_file = sample.output_folder + "rawZOBOV/" + sample.handle + "c.void"
    list_file = sample.output_folder + "rawZOBOV/" + sample.handle + "c.txt"
    vol_file = sample.output_folder + "rawZOBOV/" + sample.handle + ".trvol"
    dens_file = sample.output_folder + "rawZOBOV/" + sample.handle + ".vol"
    info_file = sample.output_folder + sample.cluster_prefix + "_cat.txt"

    # new files after post-processing
    new_clust_file = sample.output_folder + sample.cluster_prefix + ".void"
    new_list_file = sample.output_folder + sample.cluster_prefix + "_list.txt"

    # load the list of supercluster candidates
    clustersread = np.loadtxt(list_file, skiprows=2)
    # sort in desc order of max dens
    sorted_order = np.argsort(1. / clustersread[:, 3])
    clustersread = clustersread[sorted_order]

    num_clusters = len(clustersread[:, 0])
    vid = np.asarray(clustersread[:, 0], dtype=int)
    edgelist = np.asarray(clustersread[:, 1], dtype=int)
    vollist = clustersread[:, 4]
    numpartlist = np.asarray(clustersread[:, 5], dtype=int)
    rlist = clustersread[:, 9]

    # load up the cluster hierarchy
    with open(clust_file, 'r') as Fclust:
        hierarchy = Fclust.readlines()
    nclusters = int(hierarchy[0])
    if nclusters != num_clusters:
        sys.exit('Unequal void numbers in clustfile and listfile, %d and %d!' % (nclusters, num_clusters))
    hierarchy = hierarchy[1:]

    # load up the particle-zone info
    zonedata = np.loadtxt(zone_file, dtype='int', skiprows=1)

    # load the VTFE volume information
    with open(vol_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit('npart = %d in %s.trvol file does not match num_tracers = %d!'
                     % (npart, sample.handle, sample.num_tracers))
        vols = np.fromfile(File, dtype=np.float64, count=npart)

    # load the VTFE density information
    with open(dens_file, 'r') as File:
        npart = np.fromfile(File, dtype=np.int32, count=1)[0]
        if not npart == sample.num_tracers:  # sanity check
            sys.exit("npart = %d in %s.cvol file does not match num_tracers = %d!"
                     % (npart, sample.handle, sample.num_tracers))
        densities = np.fromfile(File, dtype=np.float64, count=npart)
        densities = 1. / densities

    # check whether tracer information is present, re-read in if required
    if not len(sample.tracers) == sample.num_part_total:
        sample.reread_tracer_info()
    # extract the x,y,z positions of the galaxies only (no buffer mocks)
    positions = sample.tracers[:sample.num_tracers, :3]

    # mean volume per tracer particle
    meanvol_trc = (sample.box_length ** 3.) / sample.num_part_total

    with open(new_clust_file, 'w') as Fnewclust:
        with open(new_list_file, 'w') as Fnewlist:

            # initialize variables
            counted_zones = np.empty(0, dtype=int)
            edge_flag = np.empty(0, dtype=int)
            wtd_avg_dens = np.empty(0, dtype=int)
            num__acc = 0

            for i in range(num_clusters):
                coredens = clustersread[i, 3]
                clustline = hierarchy[sorted_order[i]].split()
                pos = 1
                num_zones_to_add = int(clustline[pos])
                finalpos = pos + num_zones_to_add + 1
                rval = float(clustline[pos + 1])
                rstopadd = rlist[i]
                num_adds = 0
                if rval >= 1 and coredens > sample.max_dens_cut and numpartlist[i] >= sample.cluster_min_n \
                        and (count_all_clusters or vid[i] not in counted_zones):
                    # this zone qualifies as a seed zone
                    add_more = True
                    num__acc += 1
                    zonelist = [vid[i]]
                    total_vol = vollist[i]
                    total_num_parts = numpartlist[i]
                    zonestoadd = []
                    while num_zones_to_add > 0 and add_more:
                        zonestoadd = np.asarray(clustline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                        dens = coredens / rval
                        rsublist = rlist[np.in1d(vid, zonestoadd)]
                        volsublist = vollist[np.in1d(vid, zonestoadd)]
                        partsublist = numpartlist[np.in1d(vid, zonestoadd)]
                        if dont_merge or (use_link_density_threshold and dens < link_density_threshold) or \
                                (use_r_threshold and max(rsublist) > r_threshold):
                            # cannot add these zones
                            rstopadd = rval
                            add_more = False
                            finalpos -= (num_zones_to_add + 1)
                        else:
                            # keep adding zones
                            zonelist = np.append(zonelist, zonestoadd)
                            num_adds += num_zones_to_add
                            total_vol += np.sum(volsublist)
                            total_num_parts += np.sum(partsublist)
                        pos += num_zones_to_add + 2
                        num_zones_to_add = int(clustline[pos])
                        rval = float(clustline[pos + 1])
                        if add_more:
                            finalpos = pos + num_zones_to_add + 1

                    counted_zones = np.append(counted_zones, zonelist)
                    member_ids = np.logical_and(np.logical_or(use_stripping, densities[:] > strip_density_threshold),
                                                np.in1d(zonedata, zonelist))
                    if use_stripping:  # need to recalculate total_vol and total_num_parts after stripping
                        total_vol = np.sum(vols[member_ids])
                        total_num_parts = len(vols[member_ids])

                    if 1 in edgelist[np.in1d(vid, zonestoadd)]:
                        edge_flag = np.append(edge_flag, 1)
                    else:
                        edge_flag = np.append(edge_flag, 0)

                    # average density of member cells weighted by cell volumes
                    w_a_d = np.sum(vols[member_ids] * densities[member_ids]) / np.sum(vols[member_ids])
                    wtd_avg_dens = np.append(wtd_avg_dens, w_a_d)

                    newclustline = clustline[:finalpos]
                    if not add_more:
                        newclustline.append(str(0))
                    newclustline.append(str(rstopadd))

                    # write line to the output .void file
                    for j in range(len(newclustline)):
                        Fnewclust.write(newclustline[j] + '\t')
                    Fnewclust.write('\n')

                    if rstopadd > 10 ** 20:
                        rstopadd = -1  # will be true for structures entirely surrounded by edge particles
                    # write line to the output _list.txt file
                    Fnewlist.write('%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n' % (vid[i], int(clustersread[i, 2]), coredens,
                                                                         int(clustersread[i, 5]), num_adds + 1,
                                                                         total_num_parts, total_vol * meanvol_trc,
                                                                         rstopadd))

    # tidy up the files
    # insert first line with number of clusters to the new .void file
    with open(new_clust_file, 'r+') as Fnewclust:
        old = Fnewclust.read()
        Fnewclust.seek(0)
        topline = "%d\n" % num__acc
        Fnewclust.write(topline + old)

    # insert header to the output _list.txt file
    listdata = np.loadtxt(new_list_file)
    header = '%d non-edge tracers in %s, %d clusters\n' % (sample.num_non_edge, sample.handle, num__acc)
    header = header + 'ClusterID CoreParticle CoreDens Zone#Parts Cluster#Zones Cluster#Parts ' + \
             'ClusterVol(Mpc/h^3) ClusterDensRatio'
    np.savetxt(new_list_file, listdata, fmt='%d %d %0.6f %d %d %d %0.6f %0.6f', header=header)

    # now find the maximum density centre locations of the superclusters
    list_array = np.loadtxt(new_list_file)
    if sample.is_box:
        info_output = np.zeros((num__acc, 9))
    else:
        info_output = np.zeros((num__acc, 11))
    with open(new_clust_file, 'r') as FHierarchy:
        FHierarchy.readline()  # skip the first line, contains total number of structures
        for i in range(num__acc):
            # get the member zones of the structure
            structline = (FHierarchy.readline()).split()
            pos = 1
            add_zones = int(structline[pos]) > 0
            member_zones = np.asarray(structline[0], dtype=int)
            while add_zones:
                num_zones_to_add = int(structline[pos])
                zonestoadd = np.asarray(structline[pos + 2:pos + num_zones_to_add + 2], dtype=int)
                member_zones = np.append(member_zones, zonestoadd)
                pos += num_zones_to_add + 2
                add_zones = int(structline[pos]) > 0

            # get the member particles for these zones
            if use_stripping:
                member_ids = np.logical_and(densities[:] > strip_density_threshold, np.in1d(zonedata, member_zones))
            else:  # stripDens functionality disabled
                member_ids = np.in1d(zonedata, member_zones)
            member_vol = vols[member_ids]
            member_dens = densities[member_ids]

            # centre location is position of max. density member particle
            core_part_id = int(list_array[i, 1])
            centre = positions[core_part_id]

            # total volume of structure in Mpc/h, and effective radius
            cluster_vol = np.sum(member_vol) * meanvol_trc
            eff_rad = (3.0 * cluster_vol / (4 * np.pi)) ** (1.0 / 3)

            # average density of member cells weighted by cell volumes
            wtd_avg_dens = np.sum(member_dens * member_vol) / np.sum(member_vol)

            if sample.is_box:
                info_output[i] = [list_array[i, 0], centre[0], centre[1], centre[2], eff_rad, list_array[i, 2],
                                  wtd_avg_dens, (wtd_avg_dens - 1) * eff_rad ** 1.6, list_array[i, 7]]
            else:
                centre_obs = centre - sample.box_length / 2.0  # move back into observer coordinates
                rdist = np.linalg.norm(centre_obs)
                centre_redshift = sample.cosmo.get_redshift(rdist)
                centre_dec = 90 - np.degrees(np.arccos(centre_obs[2] / rdist))
                centre_ra = np.degrees(np.arctan2(centre_obs[1], centre_obs[0]))
                if centre_ra < 0:
                    centre_ra += 360  # to get RA in the range 0 to 360
                eff_ang_rad = np.degrees(eff_rad / rdist)
                info_output[i] = [list_array[i, 0], centre_ra, centre_dec, centre_redshift, eff_rad, list_array[i, 2],
                                  wtd_avg_dens, (wtd_avg_dens - 1) * eff_rad ** 1.6, list_array[i, 7],
                                  eff_ang_rad, edge_flag[i]]

    # save output data to file
    header = "%d superclusters from %s\n" % (num__acc, sample.handle)
    if sample.is_box:
        header = header + 'ClusterID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d %d',
                   header=header)
    else:
        header = header + 'ClusterID RA(deg) Dec(deg) redshift R_eff(Mpc/h) delta_max delta_avg lambda_c ' + \
                 'DensRatio Theta_eff(deg) EdgeFlag'
        np.savetxt(info_file, info_output, fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',
                   header=header)
