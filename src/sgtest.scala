import cc.factorie.la.DenseTensor1
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import cc.factorie.util.CmdOptions
import java.io.PrintWriter

class sgOpts extends CmdOptions {
  val embeddingFile = new CmdOption("embedding", "", "STRING", "use <string> ")
  val wordSimFile = new CmdOption("word-sim-file", "", "STRINF", "use <string>")
  val output = new CmdOption("output", "", "STRING", "use <string>")
  val threshold = new CmdOption("threshold", 0, "INT", "use <int>")
}

object sgtest {

  var vocab = Array[String]()
  var weights = Array[DenseTensor1]()
  var D = 0
  var V = 0
  var threshold = 0
   
   def main(args : Array[String]) {
      val opts = new sgOpts()
      opts.parse(args)
      val embeddingsFile = opts.embeddingFile.value
      val wordsimFile = opts.wordSimFile.value
      val outFilename = opts.output.value
      threshold = opts.threshold.value
      println(embeddingsFile)
      loadVocabAndEmbeddings(embeddingsFile)
      println(wordsimFile)
      doWordSimTask(wordsimFile, outFilename)  
   }
  
       
    private def prob(a: DenseTensor1, b: DenseTensor1): Double = 1 / (1 + math.exp(-a.dot(b)) )
    // wordsim for global sim
    private def myGlobalSim(w1: Int, w2: Int): Double = {
         val w1Embedding = weights(w1)
         val w2Embedding = weights(w2)
         return w1Embedding.dot( w2Embedding ).abs
    }
    private def myGlobalSimProb(w1: Int, w2: Int): Double = {
      val w1Embedding = weights(w1)
      val w2Embedding = weights(w2)
      return prob(w1Embedding, w2Embedding)
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

     

      val score_pred_global_mine = if (w1id != -1 && w2id != -1) myGlobalSim(w1id, w2id) else -1
      val score_pred_global_mine_prob = if (w1id != -1 && w2id != -1) myGlobalSimProb(w1id, w2id) else -1

     

      if (w1id != -1 && w2id != -1) {
        out.print("%s,%s".format(w1, w2))
        out.print(",%f".format(score_org))

        out.print(",%f".format(score_pred_global_mine))
        out.print(",%f".format(score_pred_global_mine_prob))

      
        out.print("\n")
        out.flush();
      } else not_present += 1
    }
    out.close()
    println("total not present : " + not_present)
   }
 
  def loadVocabAndEmbeddings(embeddingsFile: String): Unit = {
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
    weights = Array.ofDim[DenseTensor1](V)
    for (v <- 0 until V) {
       val line = lineItr.next.stripLineEnd
       vocab(v) = line.toUpperCase()
       val fields = lineItr.next.stripLineEnd.split(' ').map(_.toDouble)
       weights(v) = new DenseTensor1(fields)
       weights(v) /= weights(v).twoNorm
    }
    
    println("loaded vocab and their embeddings")
  }
  private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equals(word))
      return i
    return -1
  }
}