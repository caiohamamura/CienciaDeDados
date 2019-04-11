#PBS -l nodes=1:ppn=2
#PBS -l walltime=0:60:0
#PBS -o /dev/null
#PBS -e ~/error${SEED}.log

cd src/CienciaDeDados
python 02RodaNoCloudIntel.py > output.out
