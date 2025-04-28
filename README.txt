Evan Chandran, April 28, 2025

Please find the following files:

environment.py : defines parameters and settings for all models. 

model_3.py : primary technical result, global solution algorithm for decentralized equilibrium
    requires the following command line arguments:
     --tag <name of resulting directory>>
     --regime <1 or 2>
    example:
    python model_3.py --regime 1 --tag solution_dir_regime_1

plot_solo.py : plotting script for individual results
    requires the following command line arguments:
    --tag <name of directory with existing results>
    example:
    python plot_solo.py --tag solution_dir_regime_1

plot_compare.py : plotting script to compare two solutions
    requres the following command line arguments:
    --tag_a <directory of results from model a> --tag_b <directory of results from model b>
    example:
    python plot_compare.py --tag_a solution_dir_regime_1 --tag_b solution_dir_regime_2

cp_dl.py : deep learning global solution algorithm for central-planner problem
    requires the following command line arguments:
    --tag <name of resulting directory>
    example:
    python cp_dl.py --tag solution_dir_cp_dl

cp_fd.py : finite difference global solution algorithm for central-planner problem 
    requires the following command line arguments:
    --tag <name of resulting directory>
    example:
    python cp_dl.py --tag solution_dir_cp_fd

plot_cp.py : original plotting script for central planner models only [Discontinued]
    requires the following command line arguments:
    --tag <name of directory with existing results>
    example:
    python plot_solo.py --tag solution_dir_regime_1

Files in lecture_replication_and_old_model/ :
    planner_problem_implementation.ipynb : algorithm and plotting for 
        original one-state-variable problem in Appendix F

    eco529_L4_replication.ipynb : replication of lecture note model in Appendix C 
    https://github.com/goutham-fin/PrincetonECO529/blob/main/Lecture%20note%20chapters/Chapter%204.ipynb


