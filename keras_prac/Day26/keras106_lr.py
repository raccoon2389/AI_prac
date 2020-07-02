weight =0.5
input = 0.5
goal_prediction = 0.8

lr = 0.001

for itration in range(1101):
    prediction = input*weight
    error = (prediction-goal_prediction)**2

    print("Error : " + str(error)+"\tPrediction : " + str(prediction))

    up_predcition = input * (weight+lr)
    up_error = (goal_prediction - up_predcition)**2

    down_prediction = input*(weight-lr)
    down_error = (goal_prediction - down_prediction)**2

    if(down_error<up_error):
        weight = weight - lr
    
    if(down_error > up_error):
        weight = weight+lr