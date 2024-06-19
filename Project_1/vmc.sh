# Keywords: vmc, grid, int, boot, plot


if [ -z "$1" ]
then
    python3 src/simulation_scripts/vmc_playground.py
elif [ "$1" == vmc ]
then
    python3 src/simulation_scripts/vmc_playground.py
elif [ "$1" == delta_t ]
then
    python3 src/simulation_scripts/delta_t.py
elif [ "$1" == int ]
then
    python3 src/simulation_scripts/interactions.py
elif [ "$1" == boot ]
then
    python3 src/simulation_scripts/bootstrap.py
elif [ "$1" == plot ]
then 
    python3 src/simulation_scripts/plot_producer.py
elif [ "$1" == times ]
then 
    python3 src/simulation_scripts/time_measures.py
elif [ "$1" == sample ]
then
    python3 src/simulation_scripts/sample_stability.py
elif [ "$1" == landscape ]
then 
    python3 src/simulation_scripts/energy__vs_alpha.py
elif [ "$1" == one body ]
then 
    python3 src/simulation_scripts/one_body.py
else
    python3 src/simulation_scripts/"$1"
fi