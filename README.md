# ABPDB
The code is for the paper "ABPDB: A Database for Antibacterial Peptides". 
We used AlphaFold2 to predict the 3D PDB structures of 4872 peptides. These PDB data are stored in pdbs.tar, and please go to http://www.acdb.plus/ABPDB/download.php to download pdbs.tar and ABPDB.csv. As mentioned in the paper, we used the method from Can et al. to encode 4872 peptides from ABPDB into a 1280-dimensional space. The embedded vectors are saved in the file protein_data.npz. The similar_structure_search.py corresponds to the similar structure search function in http://www.acdb.plus/ABPDB, while visualization.py corresponds to the visualization function in the same site. The build_protein.py and utils.py files are from the code of Can et al.

Due to GitHub's file size limit, we have split and compressed the file protein_data.npz into protein_data.part1.rar and protein_data.part2.rar. Please download and extract both files to obtain protein_data.npz.
