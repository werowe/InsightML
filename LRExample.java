package com.bmc.ml;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;


import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.regressions.linear.LinearRegressionLSQRTrainer;
import org.apache.ignite.ml.regressions.linear.LinearRegressionModel;
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator;
import org.apache.ignite.ml.selection.scoring.metric.MetricName;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;

public class LRExample {

    public static void main(String[] args) throws IOException {
        System.out.println();
        System.out.println(">>> Linear regression model over cache based dataset usage example started.");
        // Start ignite grid.


        IgniteConfiguration igniteCfg = new IgniteConfiguration();
        igniteCfg.setWorkDirectory("/Users/walkerrowe/Downloads");
        Ignite ignite = Ignition.start(igniteCfg);


        System.out.println(">>> Ignite grid started.");


        IgniteCache<Integer, Vector> dataCache = getCache(ignite);


        try {
           // dataCache = new SandboxMLCache(ignite).fillCacheWith();


            System.out.println(">>> Create new linear regression trainer object.");
            LinearRegressionLSQRTrainer trainer = new LinearRegressionLSQRTrainer();

            System.out.println(">>> Perform the training to get the model.");


            dataCache.put(1,VectorUtils.of(1,1.8));
            dataCache.put(2,VectorUtils.of(2,4.3));
            dataCache.put(3,VectorUtils.of(3,6.2));
            dataCache.put(4,VectorUtils.of(4,5));
            dataCache.put(5,VectorUtils.of( 5,11));
            dataCache.put(6,VectorUtils.of(6,11));
            dataCache.put(7,VectorUtils.of(7,15));



            Vectorizer<Integer, Vector, Integer, Double> vectorizer = new DummyVectorizer<Integer>()
                    .labeled(Vectorizer.LabelCoordinate.FIRST);


            LinearRegressionModel mdl = trainer.fit(ignite, dataCache, vectorizer);

            double rmse = Evaluator.evaluate(
                    dataCache, mdl,
                    new DummyVectorizer<Integer>().labeled(Vectorizer.LabelCoordinate.FIRST),
                    MetricName.RMSE
            );

            System.out.println("rmse = " + rmse);

            System.out.println("intercept = " + mdl.getIntercept());


            System.out.println("Weights = " );

            Vector weights = mdl.getWeights();
            double[] w = weights.asArray();
            for (double v : w) {
                System.out.println(v);
            }




            System.out.println("==================");


        } finally {
            if (dataCache != null)
                dataCache.destroy();
        }
    }

    static private IgniteCache<Integer, Vector> getCache(Ignite ignite) {
        CacheConfiguration<Integer, Vector> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("ML_EXAMPLE_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        return ignite.createCache(cacheConfiguration);
    }





}




