# Final evaluation infos summarized
## From [MedFM Homepage](https://medfm2023.grand-challenge.org/medfm2023/)
- In the testing phase, participants would ``train their models on few-shot samples of the MedFMC public 
dataset and test performance on MedFMC reserved testing dataset``


- However, participants must ``submit the training process code in docker`` because organizers would re-run 
the training process to testify their performances.


Command to run the docker container:
````commandline
docker container run --gpus all --shm-size=8g -m 28G -it --name teamname 
--rm -v $PWD:/medfmc_exp -v $PWD/data:/medfmc_exp/data teamname:latest /bin/bash -c "sh /medfmc_exp/run.sh"
````
Basically just runs the ``run.sh``file. The currently given run.sh file contains the following code:
````commandline
export WORKDIR=/medfmc_exp
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/train.py configs/densenet/dense121_chest.py --work-dir work_dirs/temp/
````
=> If executed, trains one model and saves checkpoint to work_dirs/temp
 <br> No inference done here?

## From [MedFM Rules](https://medfm2023.grand-challenge.org/rules/)
All participants must submit a "complete solution" to this competition, including Docker container
and technical report.

"The Docker should execute for at most  3 hours and occupy no more than 10 GB GPU Memory to ``generate
prediction results`` in evaluation phase."