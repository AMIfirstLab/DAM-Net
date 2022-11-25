
def scores(C_norm):
    c = 1
    TP = C_norm[c][c]
    TN = C_norm[0][0] + C_norm[0][2]
    TN = TN + C_norm[2][0] + C_norm[2][2]
    
    FP = C_norm[0][c] + C_norm[2][c]
    FN = C_norm[c][0] + C_norm[c][2]
    
    accuracy_1 = (TP + TN) / (TP + TN + FP + FN)
    precision_1 = (TP) / (TP + FP)
    sensitivity_1 = (TP) / (TP + FN)
    specificity_1 = (TN) / (TN + FP)

    c = 2
    TP = C_norm[c][c]
    TN = C_norm[0][0] + C_norm[0][1]
    TN = TN + C_norm[1][0] + C_norm[1][1]
    
    FP = C_norm[0][c] + C_norm[1][c]
    FN = C_norm[c][0] + C_norm[c][1]
    
    accuracy_2 = (TP + TN) / (TP + TN + FP + FN)
    precision_2 = (TP) / (TP + FP)
    sensitivity_2 = (TP) / (TP + FN)
    specificity_2 = (TN) / (TN + FP)

    return accuracy_1, precision_1, sensitivity_1, specificity_1, accuracy_2, precision_2, sensitivity_2, specificity_2