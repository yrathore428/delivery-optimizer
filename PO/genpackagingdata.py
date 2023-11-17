import csv
import random

MIN_STORAGE_DAYS = 1
MAX_STORAGE_DAYS = 180
MIN_WEIGHT = 5
MAX_WEIGHT = 700
MIN_DIMENSIONS = 5
MAX_DIMENSIONS = 220
GLASS_FRAGILITY_LIMIT = 50
STORAGE_TIME_CUTOFF = 90
THRESHOLDS = {
        "Small Box": 40,
        "Medium Box": 70,
        "Large Box": 120,
        "Extra Large Box": 160,
        "Giant Box": 200,
    }

# Column headers
headers = ["length", "width", "height", "weight", "fragility", "atseal", "stime", "boxsize", "stackability", "pmaterial","" "extra_protection", "rotation"]


# options for packaging material are as follows
# cardboard box
# foam packaging
# wooden crates or pallets
# bubble wrap and air cushioning
# plastic or polyethylene box
# stretch/cling film
# anti static package
# metal casing
# vapor corrision inhibitor package
# custom made materials like thermoform trays, vacuum sealed bags, die-cut foam inserts




# Open CSV file for writing
with open("packaging_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write column headers
    writer.writerow(headers)
    
    # Generate 1000 rows
    for i in range(1000):
        
        # Generate values for each column
        length, width, height = [random.randint(MIN_DIMENSIONS, MAX_DIMENSIONS) for i in range(3)]
        weight = random.randint(MIN_WEIGHT, MAX_WEIGHT)
        
        atseal = random.randint(0,1)
        stime = random.randint(MIN_STORAGE_DAYS, MAX_STORAGE_DAYS)
        
        if(atseal == 1 and stime >= STORAGE_TIME_CUTOFF):
            pmaterial = 'plasticbox'
        
        elif(weight > 0 and weight < 50):
            if(atseal == 1):
                pmaterial= 'polyethylene'
            else:
                pmaterial = 'paperbox'
        
        elif(weight > 0 and weight < 150):
            if(atseal == 1):
                pmaterial= 'polyethylene'
            else:
                pmaterial = 'corrugated'
        
        elif(weight > 150 and weight < 400):
            if(atseal == 1):
                pmaterial= 'polyethylene'
            else:
                pmaterial = 'cardboard'
        
        elif(weight > 401 and weight < 500):
            pmaterial = 'plasticbox'
        
        elif(weight > 501 and weight < 701):
            if(atseal == 1):
                pmaterial= 'plasticbox'
            else:
                pmaterial = 'plasticwrap'
        
        fragility = random.randint(0, 1)
    
        if(fragility == 1):
            extra_protection = 1
            rotation = 1
        else:
            extra_protection = 0
            rotation = random.randint(2,3)

        if(stime > 90):
            extra_protection = 1
        
        if(extra_protection == 0):
            stackability = 1
        else:
            stackability = random.randint(0,1)


        largest_dimension = max(length, width, height)
        
        for size, threshold in THRESHOLDS.items():
            if largest_dimension <= threshold:
                boxsize = size
                break
            else:
                boxsize = "Giant Box"

        
        
        # Write row to CSV file
        writer.writerow([length, width, height, weight, fragility, atseal, stime, boxsize, stackability, pmaterial, extra_protection, rotation])
