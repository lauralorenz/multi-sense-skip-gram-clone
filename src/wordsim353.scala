import cc.factorie.la.DenseTensor1
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import java.io.PrintWriter
import cc.factorie.util.CmdOptions
import scala.collection.mutable.ArrayBuffer
import cc.factorie.la.DenseTensor


class wordSimOpts extends CmdOptions {
  val embeddingFile = new CmdOption("embedding", "", "STRING", "use <string> ")
  val output = new CmdOption("output", "", "STRING", "use <string>")
  val wordSimFile = new CmdOption("word-sim-file", "", "STRINF", "use <string>")
  val sense = new CmdOption("sense", 2, "INT", "use <int> ")
  val threshold = new CmdOption("threshold", 0, "INT", "use <int>")
  val takeSense = new CmdOption("use-sense", 1, "INT", "use <int> ")
  val dpmeans = new CmdOption("dpmeans", 0, "INT", "use dp means")
  val clusterCutOff = new CmdOption("dpmeans-cut-off", 0, "INT", "take top % for the job")
}

object wordsim353 {
   
  var vocab = Array[String]()
  var weights = Array[Array[DenseTensor1]]()
  var mu = Array[Array[DenseTensor1]]() // cluster center
  var ncluster = Array[Int]()
  var nclusterCount = Array[Array[Double]]()
  var D = 0
  var V = 0
  var S = 0
  var takeSense = false
  var threshold = 0
  var dpmeans = 0
  var dpmeansClusterCutOff = 0 
   
   def main(args : Array[String]) {
      val opts = new wordSimOpts()
      opts.parse(args)
      val embeddingsFile = opts.embeddingFile.value
      S = opts.sense.value
      val wordsimFile = opts.wordSimFile.value
      val outFilename = opts.output.value
      threshold = opts.threshold.value
      takeSense = if (opts.takeSense.value == 1) true else false // predict using sense
      dpmeans = opts.dpmeans.value.toInt
      dpmeansClusterCutOff = opts.clusterCutOff.value.toInt 
      println(embeddingsFile)
      load(embeddingsFile)
      println(wordsimFile)
      if (wordsimFile.split('/').takeRight(1)(0).startsWith("scws")) { // be careful about this.
         println("doing scws task")
         doWordSimWithContextTask(wordsimFile, outFilename)
      }
      else {
         println("doing ws-353 like task")
         doWordSimTask(wordsimFile, outFilename)
      }
      //doSpearmanCoffTest(outFilename)
      
   }
   def doSpearmanCoffTest(outFilename: String): Unit = {
         val lineItr = io.Source.fromFile(outFilename).getLines
         var scores = new ArrayBuffer[Array[Double]]
         for (line <- lineItr) {
            var v = new ArrayBuffer[Double]()
            scores.+=( line.stripLineEnd.split(',').drop(2).map(_.toDouble) ) // first two are the words
         }
         println(scores.size + " " + scores(0).size)
         val a = scores.map(row => row(0)) // true similarity score
         (1 until scores(0).size).foreach(i => {
           val b = scores.map(row => row(i)) // our model similarity score 
           println( stat.spearmanRankCoff(a, b)) //  Spearman Rank Coff
         })
   }
   def doWordSimTask(filename : String, outFilename: String): Unit =  {
    var lineItr = io.Source.fromFile(filename).getLines()
    val out = new PrintWriter(outFilename)
    var not_present = 0
    for (line <- lineItr) {
      val details = line.stripLineEnd.split(',')
      val w1 = details(0).toUpperCase
      val w2 = details(1).toUpperCase
      val score_org = details(2).toDouble
      val w1id = getID(w1)
      val w2id = getID(w2)

      val score_pred_avgsim = if (w1id != -1 && w2id != -1) avgSim(w1id, w2id) else -1
      val score_pred_avgsim_prob = if (w1id != -1 && w2id != -1) avgSimProb(w1id, w2id) else -1

      val score_pred_global_mine = if (w1id != -1 && w2id != -1) myGlobalSim(w1id, w2id) else -1
      val score_pred_global_mine_prob = if (w1id != -1 && w2id != -1) myGlobalSimProb(w1id, w2id) else -1

      val score_pred_maxsim = if (w1id != -1 && w2id != -1) maxSim(w1id, w2id) else -1
      val score_pred_maxsim_prob = if (w1id != -1 && w2id != -1) maxSimProb(w1id, w2id) else -1

      if (w1id != -1 && w2id != -1) {
        out.print("%s,%s".format(w1, w2))
        out.print(",%f".format(score_org))

        out.print(",%f".format(score_pred_global_mine))
        out.print(",%f".format(score_pred_global_mine_prob))

        out.print(",%f".format(score_pred_maxsim))
        out.print(",%f".format(score_pred_maxsim_prob))

        out.print(",%f".format(score_pred_avgsim))
        out.print(",%f".format(score_pred_avgsim_prob))

        out.print("\n")
        out.flush();
      } else not_present += 1
    }
    out.close()
    println("total not present : " + not_present)
   }
   private def prob(a: DenseTensor1, b: DenseTensor1): Double = {
          return 1 / (1 + math.exp(-a.dot(b)) )
   }
   private def logit(a: Double) = 1 / (1 + math.exp(-a))
   // hard-pick sense using contexts
   private def myLocalSim(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
         val s1 = getSense(w1, w1context)
         val s2 = getSense(w2, w2context)
         val w1s1Embedding = weights(w1)(s1)
         val w2s2Embedding = weights(w2)(s2)
         return w1s1Embedding.dot( w2s2Embedding ).abs
   }
   
   private def myLocalSimProb(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
        val s1 = getSense(w1, w1context)
        val s2 = getSense(w2, w2context)
        val w1s1Embedding = weights(w1)(s1)
        val w2s2Embedding = weights(w2)(s2)
        return prob(w1s1Embedding, w2s2Embedding)
   }
       
       
    // wordsim for global sim
    private def myGlobalSim(w1: Int, w2: Int): Double = {
         val w1Embedding = weights(w1)(0)
         val w2Embedding = weights(w2)(0)
         return w1Embedding.dot( w2Embedding ).abs
    }
    private def myGlobalSimProb(w1: Int, w2: Int): Double = {
      val w1Embedding = weights(w1)(0)
      val w2Embedding = weights(w2)(0)
      return prob(w1Embedding, w2Embedding)
    }
    // max-sim 
    private def maxSim(w1: Int, w2: Int): Double = {
          var max_score = Double.MinValue 
          val S1 = ncluster(w1); val S2 = ncluster(w2);
          for (s1 <- 1 until S1+1) for (s2 <- 1 until S2+1) {
                 val w1Embedding = weights(w1)(s1)
                 val w2Embedding = weights(w2)(s2)
                 val score = w1Embedding.dot(w2Embedding).abs * 10
                 if (score > max_score) {
                   max_score = score
                 }
           }
           max_score
    }
    private def maxSimProb(w1: Int, w2: Int): Double = {
         var max_score = Double.MinValue
          val S1 = ncluster(w1); val S2 = ncluster(w2);
         for (s1 <- 1 until S1+1) for (s2 <- 1 until S2+1) {
           val w1Embedding = weights(w1)(s1)
           val w2Embedding = weights(w2)(s2)
           val score = prob(w1Embedding, w2Embedding)
           if (score > max_score) 
             max_score = score
         }
         max_score
    }
    // soft-pick with prob(cluster, sense)
    private def avgSimC(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
             var avgSimC: Double = 0
              val c1Embedding = new DenseTensor1(D, 0)
              w1context.foreach(c => c1Embedding.+=(weights(c)(0)) )
              val c2Embedding = new DenseTensor1(D, 0)
              w2context.foreach(c => c2Embedding.+=(weights(c)(0)) )
              val p1 = new Array[Double](S+1); var z1 = 0.0;
              val p2 = new Array[Double](S+1); var z2 = 0.0; 
              val S1 = ncluster(w1); val S2 = ncluster(w2);
              for (s <- 1 until S1+1) {
                 val w1Embedding = weights(w1)(s)
                 p1(s) = prob(w1Embedding, c1Embedding)
                 z1 += p1(s)
              }
              for (s <- 1 until S2+1) {
                val w2Embedding = weights(w2)(s)
                p2(s) = prob(w2Embedding, c2Embedding)
                z2 += p2(s)
              }
              
              for (s1 <- 1 until S1+1) for (s2 <-1 until S2+1) {
                val w1Embedding = weights(w1)(s1) // 
                val w2Embedding = weights(w2)(s2) 
                avgSimC += p1(s1) * p2(s2) * w1Embedding.dot(w2Embedding).abs
              }
              return avgSimC/(1.0 * z1 * z2 * S1 * S2)
    }
   private def avgSimCProb(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
              var avgSimC: Double = 0
              val c1Embedding = new DenseTensor1(D, 0)
              w1context.foreach(c => c1Embedding.+=(weights(c)(0)) )
              val c2Embedding = new DenseTensor1(D, 0)
              w2context.foreach(c => c2Embedding.+=(weights(c)(0)) )
              val p1 = new Array[Double](S+1); var z1 = 0.0;
              val p2 = new Array[Double](S+1); var z2 = 0.0;
              val S1 = ncluster(w1); val S2 = ncluster(w2);
              for (s <- 1 until S1+1) {
                 val w1Embedding = weights(w1)(s)
                 p1(s) = prob(w1Embedding, c1Embedding)
                 z1 += p1(s)
              }
              for (s <- 1 until S2+1) {
                val w2Embedding = weights(w2)(s)
                p2(s) = prob(w2Embedding, c2Embedding)
                z2 += p2(s)
              }
              
              for (s1 <- 1 until S1+1) for (s2 <-1 until S2+1) {
                val w1Embedding = weights(w1)(s1) // w1 embedding
                val w2Embedding = weights(w2)(s2) // w2 embedding 
                val pb = prob(w1Embedding, w2Embedding)
                print("[" + p1(s1)/z1 + "," + p2(s2)/z2 + "," + pb + "] ")
                avgSimC += p1(s1) * p2(s2) * pb
              }
              print("\n")
              return avgSimC/(1.0 * z1 * z2)
    }
    private def avgSimCSocher(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
              var avgSimC: Double = 0
              val c1Embedding = new DenseTensor1(D, 0)
              w1context.foreach(c => c1Embedding.+=(weights(c)(0)) )
              val c2Embedding = new DenseTensor1(D, 0)
              w2context.foreach(c => c2Embedding.+=(weights(c)(0)) )
              val p1 = new Array[Double](S+1); var z1 = 0.0;
              val p2 = new Array[Double](S+1); var z2 = 0.0; 
              val S1 = ncluster(w1); val S2 = ncluster(w2);
              for (s <- 1 until S1+1) {
                 val w1Embedding = weights(w1)(s)
                 //p1(s) = prob(mu(w1)(s-1), c1Embedding)
                 p1(s) = 1.0/(1.0 - TensorUtils.cosineDistance(mu(w1)(s-1), c1Embedding))
                // p1(s) = math.exp(TensorUtils.cosineDistance(mu(w1)(s-1), c1Embedding)) //.abs // proportional to the distance
                 z1 += p1(s) // normalzing constant
              }
              for (s <- 1 until S2+1) {
                val w2Embedding = weights(w2)(s)
                //p2(s) = prob(mu(w2)(s-1), c2Embedding)
                p2(s) = 1.0/(1.0 - TensorUtils.cosineDistance(mu(w2)(s-1), c2Embedding))
                //p2(s) = math.exp(TensorUtils.cosineDistance(mu(w2)(s-1), c2Embedding)) //.abs
                z2 += p2(s)
              }
              
              for (s1 <- 1 until S1+1) for (s2 <-1 until S2+1) {
                val w1Embedding = weights(w1)(s1) // 
                val w2Embedding = weights(w2)(s2) 
                avgSimC += p1(s1) * p2(s2) *prob(w1Embedding, w2Embedding)
               // val diff = 
               // avgSimC += p1(s1) * p2(s2) * prob(mu(w1)(s1-1), mu(w2)(s2-1))
              }
              return avgSimC/(1.0 * z1 * z2)
    }
   private def avgSimCDist(w1: Int, w2: Int, w1context: Seq[Int], w2context: Seq[Int]): Double = {
              var avgSimC: Double = 0
              val c1Embedding = new DenseTensor1(D, 0)
              w1context.foreach(c => c1Embedding.+=(weights(c)(0)) )
              //c1Embedding /= c1Embedding.twoNorm
              val c2Embedding = new DenseTensor1(D, 0)
              w2context.foreach(c => c2Embedding.+=(weights(c)(0)) )
              //c2Embedding /= c2Embedding.twoNorm
              val p1 = new Array[Double](S+1); var z1 = 0.0;
              val p2 = new Array[Double](S+1); var z2 = 0.0; 
              val S1 = ncluster(w1); val S2 = ncluster(w2);
              for (s <- 1 until S1) {
                 val w1Embedding = weights(w1)(s)
                // val dist1 = prob(w1Embedding, c1Embedding)
                 val dist1 = w1Embedding.dot(c1Embedding)
                 p1(s) = dist1
                 z1 += p1(s)
              }
              for (s <- 1 until S2) {
                val w2Embedding = weights(w2)(s)
                //val dist2 = prob(w2Embedding, c2Embedding)
                val dist2 = w2Embedding.dot(c2Embedding)
                p2(s) = dist2
                z2 += p2(s)
              }
              // z1 -> normalize the distance
              // z2 -> normalize the distance
              for (s1 <- 1 until S1) for (s2 <-1 until S2) {
                val w1Embedding = weights(w1)(s1) // 
                val w2Embedding = weights(w2)(s2) 
                avgSimC += p1(s1) * p2(s2) * prob(w1Embedding, w2Embedding)
              }
              return avgSimC/(1.0 * z1 * z2 * S1 * S2)
    }
    // soft-pick with equal prob to all sense
    def avgSim(w1: Int, w2: Int): Double = {
             var avgSim:Double = 0
             val S1 = ncluster(w1); val S2 = ncluster(w2);
             for (s1 <- 1 until S1) for (s2 <- 1 until S2) {
                val w1s1 = weights(w1)(s1) 
                val w2s2 = weights(w2)(s2) 
                avgSim += w1s1.dot(w2s2).abs
             }
             return avgSim/(S1*S2)
    }
    def avgSimProb(w1:Int, w2:Int): Double = {
         var avgSim: Double = 0
         val S1 = ncluster(w1); val S2 = ncluster(w2);
         for (s1 <- 1 until S1+1) for (s2 <- 1 until S2+1) {
           val w1Embedding = weights(w1)(s1)
           val w2Embedding = weights(w2)(s2)
           avgSim += prob(w1Embedding, w2Embedding)
         }
         return avgSim/(S1*S2)
    }
    
    def doWordSimWithContextTask(filename: String, outFilename: String): Unit = {
    var lineItr = io.Source.fromFile(filename).getLines()
    val out = new PrintWriter(outFilename)
    var not_present = 0
    while (lineItr.hasNext) {
      val details   = lineItr.next.stripLineEnd.split(',') // <word1><comma><word2><comma><score>
      var context = lineItr.next.stripLineEnd.split(' ')
      val w1context = context.map(word => word.toUpperCase).map(word => getID(word)).filter(id => id != -1) // filters contexts
      context = lineItr.next.stripLineEnd.split(' ')
      val w2context = context.map(word => word.toUpperCase).map(word => getID(word)).filter(id => id != -1) // filterd contexts
      val w1 = details(0).toUpperCase
      val w2 = details(1).toUpperCase
      val score_org = details(2).toDouble
      val w1id = getID(w1) // get word id
      val w2id = getID(w2) // get word id
      var w1sense = 0
      var w2sense = 0
      if (takeSense && w1id != -1 && w2id != -1) { // use the contexts only take_sense = 1
        w1sense = getSense(w1id, w1context)
        w2sense = getSense(w2id, w2context)
      }
     //  val score_pred_avgsimc = if (w1id != -1 && w2id != -1 ) avgSimC(w1id, w2id, w1context, w2context) else -1
     //  val score_pred_avgsimc_prob = if (w1id != -1 && w2id != -1 ) avgSimCProb(w1id, w2id, w1context, w2context) else -1
     // val score_pred_avgsimc_dist = if (w1id != -1 && w2id != -1) avgSimCDist(w1id, w2id, w1context, w2context) else -1
      
     // val score_pred_avgsim = if (w1id != -1 && w2id != -1) avgSim(w1id, w2id) else -1
      val score_pred_avgsim_prob = if (w1id != -1 && w2id != -1) avgSimProb(w1id, w2id) else -1
       
     // val score_pred_local_mine = if (w1id != -1 && w2id != -1) myLocalSim(w1id, w2id, w1context, w2context) else -1
      val score_pred_local_mine_prob = if (w1id != -1 && w2id != -1) myLocalSimProb(w1id, w2id, w1context, w2context) else -1
       
    //  val score_pred_global_mine = if (w1id != -1 && w2id != -1) myGlobalSim(w1id, w2id) else -1
      val score_pred_global_mine_prob = if (w1id != -1 && w2id != -1) myGlobalSimProb(w1id, w2id) else -1
      
    //  val score_pred_maxsim = if (w1id != -1 && w2id != -1) maxSim(w1id, w2id) else -1
     val score_pred_maxsim_prob = if (w1id != -1 && w2id != -1) maxSimProb(w1id, w2id) else -1
      
      var score_pred_avgsimc_socher = -1.0
      score_pred_avgsimc_socher = if (dpmeans == 1 &&  w1id != -1 && w2id != -1) avgSimCSocher(w1id, w2id, w1context, w2context) else -1.0
      
      if (w1id  != -1 && w2id != -1) { 
          out.print("%s,%s".format(w1, w2))
          out.print(",%f".format(score_org))
          
         
           
        //  out.print(",%f".format(score_pred_local_mine))
          out.print(",%f".format(score_pred_local_mine_prob)) 
          
       //   out.print(",%f".format(score_pred_global_mine)) 
          out.print(",%f".format(score_pred_global_mine_prob))
          
          //out.print(",%f".format(score_pred_maxsim))
         out.print(",%f".format(score_pred_maxsim_prob))
          
  //        out.print(",%f".format(score_pred_avgsim))
          out.print(",%f".format(score_pred_avgsim_prob))
          
       //   out.print(",%f".format(score_pred_avgsimc)) 
       //   out.print(",%f".format(score_pred_avgsimc_prob))
      //    out.print(",%f".format(score_pred_avgsimc_dist))
          if (dpmeans == 1) {
            out.print(",%f".format(score_pred_avgsimc_socher))
          }
          
          out.print("\n")
          out.flush() ;
      }
      else not_present += 1
    }
    println("total not present : " + not_present)
  }
   private def getSense(word : Int, contexts : Seq[Int]): Int =  {
        val contextEmbedding = new DenseTensor1(D, 0)
        // contexts.foreach(context)
        (0 until contexts.size).foreach(i => contextEmbedding.+=(weights(contexts(i))(0)) ) // global context
        var correct_sense = 0
        var max_score = Double.MinValue
        for (s <- 1 until ncluster(word)+1) {
             val score = contextEmbedding.dot( weights(word)(s) )// find the local context
             if (score > max_score) {
               correct_sense = s
               max_score = score
             }
        }
        correct_sense
  }
    def load(embeddingsFile: String): Unit = {
     var lineItr = embeddingsFile.endsWith(".gz") match {
      case false => io.Source.fromFile(embeddingsFile).getLines
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(embeddingsFile)), "iso-8859-1").getLines
    }
     // first line is (# words, dimension)
     val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
     V = if (threshold > 0 && details(0) > threshold) threshold else details(0)
     D = details(1)
     println("# words : %d , # size : %d".format(V, D))
     vocab = new Array[String](V)
     weights = Array.ofDim[DenseTensor1](V, S+1)
     mu = Array.ofDim[DenseTensor1](V, S)
     ncluster = Array.ofDim[Int](V)
     nclusterCount = Array.ofDim[Double](V, S)
     for (v <- 0 until V) {
      val line = lineItr.next.stripLineEnd.split(' ')
      vocab(v) = line(0).toUpperCase
      if (line.size > 1) 
        ncluster(v) = line(1).toInt // # of senses for the word it learnt
      else ncluster(v) = S
      
      if (dpmeans == 1 && line.size > 2) {
            var totalClusterCount = 0.0
            for (s <- 2 until line.size) {
              nclusterCount(v)(s-2) = line(s).toInt
              totalClusterCount += nclusterCount(v)(s-2)
            }
            for (s <- 2 until line.size)
               nclusterCount(v)(s-2) = (100 * nclusterCount(v)(s-2)) / (1.0 * totalClusterCount)
      }
      
      if (dpmeans == 1) {
          var i = 0
          for (s <- 0 until ncluster(v)+1) {
            val fields = lineItr.next.stripLineEnd.split(' ').map(_.toDouble)
            if (s == 0) {
             weights(v)(0) = new DenseTensor1(fields)
             weights(v)(0) /= weights(v)(0).twoNorm
            }
            if (s > 0){
              val f = lineItr.next.stripLineEnd.split(' ').map(_.toDouble)
              if (nclusterCount(v)(s-1) > dpmeansClusterCutOff) {
                i += 1
                weights(v)(i) = new DenseTensor1(fields)
                weights(v)(i) /= weights(v)(i).twoNorm
              
                mu(v)(i-1) = new DenseTensor1(f)
                mu(v)(i-1) /= mu(v)(i-1).twoNorm
              }
            }
          }
          ncluster(v) = i
      }
      else {
        for (s <- 0 until ncluster(v)+1) {
        val fields = lineItr.next.stripLineEnd.split(' ').map(_.toDouble)
        weights(v)(s) = new DenseTensor1(fields)
        weights(v)(s) /= weights(v)(s).twoNorm
        // load the MUs only if dpmeans flag is set
        if (dpmeans == 1 && s > 0) {
             val f = lineItr.next.stripLineEnd.split(' ').map(_.toDouble)
             mu(v)(s-1) = new DenseTensor1(f)
             mu(v)(s-1) /= mu(v)(s-1).twoNorm
        }
      }
      }
      if (v%50000 == 0) println("loaded " + v + " words")
     }
    println("loaded vocab and their embeddings")
  }
    private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equals(word))
      return i
    return -1
  }
}
