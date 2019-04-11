#PBS -l nodes=4
#PBS -l walltime=0:10:0
#PBS -o /dev/null
#PBS -e ~/error${SEED}.log

cd src/CienciaDeDados
mpirun python 02TensorFlor.py > output.out
