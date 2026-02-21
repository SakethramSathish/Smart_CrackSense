import numpy as np

def normalize(value, max_value):
    return min(value / max_value, 1.0)

def compute_severity_index(length, width, density, num_cracks):
    #No crack condition
    if num_cracks == 0:
        return 0.0
    
    #Empirical normalization constants
    L_norm = normalize(length, 800)
    W_norm = normalize(width, 8)
    D_norm = normalize(density, 0.08)

    if density < 0.002:
        return 0.0
    
    print("Length:", length)
    print("Width:", width)
    print("Density:", density)
    print("Num cracks:", num_cracks)

    SI = 0.4 * L_norm + 0.4 * W_norm + 0.2 * D_norm

    return SI * 5   #Scale to 0-5 range

def classify_severity(SI):
    """
    Maps severity index to class.
    """

    if SI == 0:
        return "No Crack"
    elif SI < 1.5:
        return "Minor"
    elif SI < 3:
        return "Moderate"
    else:
        return "Severe"