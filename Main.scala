package com.bmc.ignite

import org.apache.ignite.Ignite
import org.apache.ignite.Ignition
import org.apache.ignite.configuration.IgniteConfiguration
import java.io.IOException
import java.util
import java.util.UUID
import org.apache.ignite.Ignite
import org.apache.ignite.IgniteCache
import org.apache.ignite.Ignition
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction
import org.apache.ignite.configuration.CacheConfiguration
import org.apache.ignite.configuration.IgniteConfiguration
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer
import org.apache.ignite.ml.math.primitives.vector.Vector
import org.apache.ignite.ml.math.primitives.vector.VectorUtils
import org.apache.ignite.ml.regressions.linear.LinearRegressionLSQRTrainer
import org.apache.ignite.ml.regressions.linear.LinearRegressionModel
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator
import org.apache.ignite.ml.selection.scoring.metric.MetricName



object Main extends App {


  val PersistencePath = "/Users/walkerrowe/Downloads/ignite"
  val WalPath = "/Users/walkerrowe/Downloads/wal"

  val config = new IgniteConfiguration()


  val ignite = Ignition.start(config)




  val dataCache = getCache(ignite)



  dataCache.put(1, VectorUtils.of(1, 1.8))
  dataCache.put(2, VectorUtils.of(2, 4.3))
  dataCache.put(3, VectorUtils.of(3, 6.2))
  dataCache.put(4, VectorUtils.of(4, 5))
  dataCache.put(5, VectorUtils.of(5, 11))
  dataCache.put(6, VectorUtils.of(6, 11))
  dataCache.put(7, VectorUtils.of(7, 15))

  println("data created")


  val vectorizer = new DummyVectorizer[Integer]().labeled(Vectorizer.LabelCoordinate.FIRST)

  val trainer = new LinearRegressionLSQRTrainer

  val mdl = trainer.fit(ignite, dataCache, vectorizer)

  val rmse = Evaluator.evaluate(dataCache, mdl, new DummyVectorizer[Integer]().labeled(Vectorizer.LabelCoordinate.FIRST), MetricName.RMSE)

  print("rmse" + rmse)

  System.out.println("intercept = " + mdl.getIntercept)


  System.out.println("Weights = ")

  val weights = mdl.getWeights
  val w = weights.asArray
  for (v <- w) {
    System.out.println(v)
  }




  private def getCache(ignite: Ignite) = {
    val cacheConfiguration = new CacheConfiguration[Integer, Vector]
    cacheConfiguration.setName("ML_EXAMPLE_" + UUID.randomUUID)
    cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10))
    ignite.createCache(cacheConfiguration)
  }


}