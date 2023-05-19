import csv
import random

MIN_STORAGE_DAYS = 1
MAX_STORAGE_DAYS = 180
MIN_WEIGHT = 1
MAX_WEIGHT = 20
MIN_DIMENSIONS = 5
MAX_DIMENSIONS = 100
GLASS_FRAGILITY_LIMIT = 50
STORAGE_TIME_CUTOFF = 90

# Column headers
headers = ["length", "width", "height", "weight", "fragility", "atseal", "stime", "alloy", "plastic", "glass", "pmaterial", "pthickness_class", "extra_protection", "pfragility"]

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
        if(atseal == 1 or stime >= STORAGE_TIME_CUTOFF):
            pmaterial = 'plasticbox'
            pthickness_class = 2
        elif(weight > 0 and weight < 3):
            pmaterial = 'plasticwrap'
            pthickness_class = 0
        elif(weight > 2 and weight < 5):
            pmaterial= 'paperbox'
            pthickness_class = 0
        elif(weight > 4 and weight < 10):
            pmaterial= 'corrugated'
            pthickness_class = 1
        elif(weight > 9 and weight < 15):
            pmaterial= 'plasticbox'
            pthickness_class = 1
        elif(weight > 14 and weight < 21):
            pmaterial= 'cardboard'
            pthickness_class = 2
        
        fragility = random.randint(1, 5)
    
        if(fragility > 3):
            pfragility = 2
            extra_protection = 1
        elif(fragility > 1 and fragility < 4):
            pfragility = 1
            extra_protection = 1
    
        else:
            pfragility = 0
            extra_protection = 0

        if(stime > 90):
            extra_protection = 1
            pfragility = 1
        
        alloy, plastic, glass = random.sample(range(1, 100), 3)
        
        # Check if sum of first three columns adds up to 100
        while alloy + plastic + glass != 100:
            alloy, plastic, glass = random.sample(range(1, 100), 3)
        
        if(glass >= GLASS_FRAGILITY_LIMIT):
            extra_protection = 1
            pfragility = 1
        elif(glass < GLASS_FRAGILITY_LIMIT and pfragility == 1):
            pfragility = 1
            extra_protection = 1
        elif(glass < GLASS_FRAGILITY_LIMIT and pfragility == 0):
            pfragility = 0
            extra_protection = 0
        # Write row to CSV file
        writer.writerow([length, width, height, weight, fragility, atseal, stime, alloy, plastic, glass, pmaterial, pthickness_class, extra_protection, pfragility])
