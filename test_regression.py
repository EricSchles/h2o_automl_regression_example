import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "x1"
x.remove("response")
x.remove(y)

# Run AutoML for 30 seconds
# This is how AutoML chooses whether to do regression or classification
# classification by default, regression if stopping_metric="deviance"
aml = H2OAutoML(max_runtime_secs = 30, stopping_metric="deviance")
aml.train(x = x, y = y,
          training_frame = train,
          leaderboard_frame = test)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)


# The leader model is stored here
print(aml.leader)

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)
