import subprocess

subprocess.call(['python','./src/model/A_StructureInputFiles.py'])
print "A complete"
subprocess.call(['python','./src/model/B_Forecast.py'])
print "B complete"
subprocess.call(['python','./src/model/C_DeriveColumns.py'])
print "C complete"
subprocess.call(['python','./src/model/D_Regress.py'])
print "D complete"
subprocess.call(['python','./src/model/E_LWS_Step.py'])
print "E complete"
subprocess.call(['python','./src/model/F_StdDev_LWS.py'])
print "F complete"
subprocess.call(['python','./src/model/G_MicroRegress.py'])
print "G complete"
subprocess.call(['python','./src/model/H_CombineModels.py'])
print "Done"
