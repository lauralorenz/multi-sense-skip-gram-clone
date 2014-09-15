import cc.factorie.la.DenseTensor1
import scala.io.Source
import java.util.zip.GZIPInputStream
import java.io.FileInputStream

object accuracy {
  var threshold = 0
   var vocab = Array[String]()
  var weights = Array[Array[DenseTensor1]]()
  var mu = Array[Array[DenseTensor1]]() // cluster center
  var ncluster = Array[Int]()
  var nclusterCount = Array[Array[Double]]()
  var D = 0
  var V = 0
  var dpmeans = 1
  var dpmeansClusterCutOff = 0
  var S = 0
  var G = true // predict using global contexts 
  def main(args: Array[String]) {
    val inputFile = args(0)
    val testFile = args(4)
    threshold = args(2).toInt
    S = args(1).toInt
    G = if (args(3).toInt == 0) false else true
    load(inputFile)
    test(testFile)
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
  def test(filename : String): Unit = {
        var curr_task_correct = 0
        var curr_task_total = 0
        var sem_correct = 0
        var sem_total = 0
        var syn_correct = 0
        var syn_total = 0
        var all_correct =0
        var all_total = 0
        var ntask = 0
        for (line <- io.Source.fromFile(filename).getLines) {
          if (line.startsWith(":")) {
              if (curr_task_total > 0) {
                if (ntask > 5) {
                  syn_correct += curr_task_correct
                  syn_total += curr_task_total
                }
                else {
                  sem_correct += curr_task_correct
                  sem_total += curr_task_total
                }
                all_correct += curr_task_correct 
                all_total += curr_task_total
                 println("ACCURACY TOP1: " + 100*curr_task_correct/curr_task_total.toDouble + " % " + " (" + 
                    curr_task_correct + " / " + curr_task_total + ")")
                 print("Total accuracy: " + 100*all_correct/all_total.toDouble + " % ")
                 print(" Semantic accuracy: " + 100*sem_correct/sem_total.toDouble + " % ")
                 println(" Syntactic accuracy: " + 100*syn_correct/syn_total.toDouble + " %")
              }
              curr_task_total = 0
              curr_task_correct = 0
              ntask += 1
              println("Task" + line)
              
          }
          else {
            val words = line.stripLineEnd.split(' ').map(word => word.toUpperCase).map(word => getID(word)).filter(id => id != -1)
            if (words.size == 4) {
                   
                   var pred_v : Int = 0
                   var correct_v: Int = words(3)
                   val given_words = words.take(3)
                   if (G == true) { 
                     val found_senses = getSenses(given_words, global = G)
                     pred_v = closest_vec(given_words, found_senses, global = G)
                   }
                   // use multiple embeddings
                   else {
                       pred_v = closest_vec_avgsim(given_words(0), given_words(1), given_words(2))
                   }
                  if (vocab(pred_v).equals( vocab(correct_v)))
                       curr_task_correct += 1
                   curr_task_total += 1
                  
            }
            
          }
        }
        println("ACCURACY TOP1: " + 100*curr_task_correct/curr_task_total.toDouble + " % " + " (" + 
                    curr_task_correct + " / " + curr_task_total + ")")
                 print("Total accuracy: " + 100*all_correct/all_total.toDouble + " % ")
                 print(" Semantic accuracy: " + 100*sem_correct/sem_total.toDouble + " % ")
                 println(" Syntactic accuracy: " + 100*syn_correct/syn_total.toDouble + " %")
  }
  private def getSenses(words: Seq[Int], global : Boolean = true): Seq[Int] = {
           val senses = new Array[Int](words.size)
           for (i <- 0 until words.size) {
             val w = words(i)
             val contexts = words.filter(id => id != w)
             senses(i) = if (global) 0 else getSense(w, contexts)
           }
           senses
  }
  private def getSense(word : Int, contexts : Seq[Int]): Int =  {
        val contextEmbedding = new DenseTensor1(D, 0)
        // contexts.foreach(context)
        (0 until contexts.size).foreach(i => contextEmbedding.+=(weights(contexts(i))(0)) ) // global context
        var correct_sense = 0
        var max_score = Double.MinValue
        for (s <- 1 until S+1) {
             val score = contextEmbedding.dot( weights(word)(s) )// find the local context
             if (score > max_score) {
               correct_sense = s
               max_score = score
             }
        }
        correct_sense
  }
  private def closest_vec_avgsim(a : Int, b: Int, c : Int): Int = {
        var max_score = Double.MinValue
        var ans = -1
        for (sa <- 1 until ncluster(a) +1 ) 
        {
          for (sb <- 1 until ncluster(b)+1) 
          {
            for (sc <- 1 until ncluster(c)+1 ) 
            {
              val in = weights(c)(sc) + weights(b)(sb) - weights(a)(sa)
              for (d <- 0 until V) 
              {
                var score = 0.0
                var Z = ncluster(a) * ncluster(b) * ncluster(c) * ncluster(d)
                for (sd <- 1 until ncluster(d) +1) 
                {
                     val out = weights(d)(sd)
                     val dot = in.dot(out)
                     score += prob(dot) * TensorUtils.cosineDistance(in, out)
                }
                score = score/(1.0 * Z)
                if (score > max_score) 
                {
                  max_score = score
                  ans = d
                }
              }
            }
          }
        }
        ans
             
  }
   private def closest_vec(words: Seq[Int], senses : Seq[Int], global : Boolean = true): Int = {
         val in = new DenseTensor1(D, 0)
         in.-=( weights(words(0))(senses(0))  )
         in.+=( weights(words(1))(senses(1))  )
         in.+=( weights(words(2))(senses(2))  )
         var max_score = Double.MinValue
         var ans = -1
         var start = 0
         var end = 0
         if (global == true) {
           start = 0
           end = 1
         }
         else {
           start = 1
           end = S + 1
         }
           for (v <- 0 until V) {
             for (s <- start until end) {
             val out = weights(v)(s)
             val score = in.dot(out)
             if (score > max_score && v != words(0) && v != words(1) && v != words(2)) {
               ans = v
               max_score = score
             }
             }
           }
           ans
   }
   private def prob(x : Double): Double = 1 / (1 + math.exp(-x))
    
   private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equals(word))
      return i
    return -1
  }
}