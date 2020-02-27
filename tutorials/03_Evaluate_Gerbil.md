# Evaluate on GERBIL
For evaluation we used [GERBIL](https://github.com/dice-group/gerbil/). GERBIL requires middleware software, which we have provided in our scripts
folder. The first thing that needs to be done is forwarding a port that GERBIL
can refer to. This port should refer to the port 1235, which we use for our middleware code. Alternatively, one could choose to download the local GERBIL platform, but we could not get that to
work properly.

After opening the `gerbil_middleware` folder, the user can run the following command to start the middleware:
```
mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```
Secondly, we need to enable the API as described in the previous tutorials. Here the respective
modes are either 'ED' or 'EL', depending on what the user wishes to evaluate.

Finally, the user may open the [GERBIL platform](http://gerbil.aksw.org/gerbil/config) and configure and run an experiment.