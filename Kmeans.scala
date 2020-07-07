
package com.bmc.ignite


import java.util.UUID

import org.apache.ignite.Ignite

import org.apache.ignite.Ignition
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction
import org.apache.ignite.configuration.CacheConfiguration
import org.apache.ignite.configuration.IgniteConfiguration
import org.apache.ignite.ml.clustering.kmeans.{KMeansModel, KMeansTrainer}


import org.apache.ignite.ml.math.primitives.vector.Vector
import org.apache.ignite.ml.math.primitives.vector.VectorUtils

import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer

import org.apache.ignite.cache.query.ScanQuery


object Main extends App {

  val PersistencePath = "/Users/walkerrowe/Downloads/ignite"
  val WalPath = "/Users/walkerrowe/Downloads/wal"

  val config = new IgniteConfiguration()

  val ignite = Ignition.start(config)

  val dataCache = getCache(ignite)

  val file = "/Users/walkerrowe/Documents/igniteSource/ignite/examples/src/main/resources/datasets/two_classed_iris.csv"

  val bufferedSource = io.Source.fromFile(file)

  var i: Integer = 0
  for (line <- bufferedSource.getLines) {

    val cols: Array[String] = line.split("\\s+").map(_.trim)

    dataCache.put(i, VectorUtils.of(cols(0).toDouble,
      cols(1).toDouble, cols(2).toDouble,
      cols(3).toDouble, cols(4).toDouble))
    i = i + 1
  }
  bufferedSource.close

  val vectorizer = new DummyVectorizer[Integer]().labeled(Vectorizer.LabelCoordinate.FIRST)

  val trainer = new KMeansTrainer()
  trainer.withAmountOfClusters(2)

  val mdl = trainer.fit(ignite, dataCache, vectorizer)

  val centers: Array[Vector] = mdl.getCenters

  for (c <- centers) {
    System.out.print("centers ")
    for (a <- c.asArray()) {
      printf("%.2f ", a)
    }
    System.out.println("\n")
  }

  val cursor = dataCache.query(new ScanQuery[Int, Vector])
  val all = cursor.getAll


  for (i <- 0 until all.size() - 1) {
    var r = all.get(i)

    val features = r.getValue.copyOfRange(1, r.getValue.size())
    val label = r.getValue.get(0)
    printf("label=%.2f = features=", label)

    for (f <- features.asArray()) {
      printf("%.2f,", f)
    }
    print("\n")

    val prediction = mdl.predict(features)


    System.out.printf("prediction = %.2f \n", prediction.toDouble)

    println("===========================================")

  }


  private def getCache(ignite: Ignite) = {
    val cacheConfiguration = new CacheConfiguration[Integer, Vector]
    cacheConfiguration.setName("ML_EXAMPLE_" + UUID.randomUUID)
    cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10))
    ignite.createCache(cacheConfiguration)
  }

}



