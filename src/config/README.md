#Creating New Config Files


###UPDATED 2015-10-29

To explore your own set of hyperparameters, you can define your own config files. They must conform to the following rules:

1. They must have a `selectors` and `learners` dictionary.
2. The key of each dictionary entry must be a tuple in the form (name * object). 
  The object is a transform or estimator. These are classes implement the `fit` and `transform` methods.
  The name is a unique name that represents the object. 
3. The values of each dictionary entry must be themselves a dictionary. These dictionaries represent parameters to their resepective transform or estimator objects. Each of these dictionaries must have keys of the form "`objectname__paramname`" where object name is the name you give to the corresponding object, and param name is the name of the object's parameter. **NOTE THAT THERE MUST BE TWO UNDERSCORES DELIMITING `objectname` and `paramname`**
