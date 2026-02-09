1. Installer simgrid.4.0 (https://simgrid.org/)

2. Positionner les variables d'environnements pour accéder au compilateur (smpicc) et au script d'exécution parallèle (smpirun)

3. Compiler (make)

4. Tester

smpirun -np 4 -hostfile cluster_hostfile.txt -platform cluster_crossbar.xml ./hello_world

affichage attendu

des messages de SimGrid

Hello world from process 0 of 4
Hello world from process 1 of 4
Hello world from process 2 of 4
Hello world from process 3 of 4
