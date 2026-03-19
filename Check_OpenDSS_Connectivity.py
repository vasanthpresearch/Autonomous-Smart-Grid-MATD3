import opendssdirect as dss
dss.Basic.Start()
dss.Text.Command('new circuit.Test phases=3 bus1=SourceBus basekv=11')
dss.Solution.Solve()
print(f"Grid Voltage: {dss.Circuit.AllBusMagPu()[0]:.4f} pu") 
# If this prints ~1.0000, your OpenDSS engine is working.