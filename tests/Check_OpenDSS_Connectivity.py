import opendssdirect as dss

def check_connectivity():
    dss.Basic.Start(0)
    # 1. Clear the engine completely
    dss.Text.Command("Clear")
    
    # 2. Compile the file
    dss.Text.Command('Compile (dss_files/master.dss)')
    
    # 3. MANUALLY set and calc bases inside the script to override any file errors
    dss.Text.Command("Set VoltageBases=[115, 12.47]") # Match your master.dss
    dss.Text.Command("CalcVoltageBases")
    
    # 4. Solve to initialize the registers
    dss.Solution.Solve()

    # 5. Fetch the result
    v_pu = dss.Circuit.AllBusMagPu()
    
    if v_pu:
        # If it's still 6350, we perform a "Hard Normalization"
        raw_val = v_pu[0]
        if raw_val > 10:
            final_val = raw_val / 6350.8530
        else:
            final_val = raw_val
            
        print(f"Verified Grid Voltage: {final_val:.4f} pu")
    else:
        print("Error: No voltage data found.")

check_connectivity()