year='Y1'
# Loa Iron Iron_blinded
pipeline='Iron_blinded'

# load DESI environment
source /global/common/software/desi/desi_environment.sh main
#module load openmpi


#y3
#declare -a tracers=('QSO' 'LRG' 'ELG' 'BGS' )
#declare -a tracers_exact=('QSO' 'LRG' 'ELGnotqso' 'BGS_ANY')
#declare -a caps=('NGC' 'SGC')

#y1
declare -a tracers=('QSO' 'LRG' 'ELG' 'BGS' )
declare -a tracers_exact=('QSO' 'LRG' 'ELG_LOPnotqso' 'BGS_BRIGHT-21.5')
declare -a caps=('NGC' 'SGC')


# number of tracers
numtracers=${#tracers[@]}


for (( i=0; i<${numtracers}; i++ ));
do
    tracer=${tracers[$i]}
    tracer_exact=${tracers_exact[$i]}
    for cap in "${caps[@]}"
    do
        cd ./parameters
       
        # set up revovler param file
        python edit_params_cutsky_data.py --year $year --pipeline $pipeline --tracer $tracer --tracer2 $tracer_exact --cap $cap --algorithm voxel
        
        cd ..
    
        # run the REVOLVER void-finder
        # maybe replace tags with $SLURM_JOB_ID (would need to pass $SLURM_JOB_ID to edit_params_cutsky_data)
        python revolver.py --par parameters/params_cutsky_"$year"_"$tracer"_"$pipeline".py
    
    done
done



## loop through above array







# remove params_cutsky_$year_$tracer_$pipeline.py
#rm parameters/params_cutsky_"$year"_"$tracer"_"$pipeline".py

# clear out all files in tmp folder
#rm -rf /pscratch/sd/h/hrincon/desigroup/voids/tmp/*
