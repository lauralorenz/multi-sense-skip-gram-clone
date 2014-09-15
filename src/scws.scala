import cc.factorie.la.DenseTensor1
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import cc.factorie.util.CmdOptions

class SCWSOpts extends CmdOptions {
  val embeddingFile = new CmdOption("embedding", "", "STRING", "use <string> ")
  val wordSimFile = new CmdOption("word-sim-file", "", "STRINF", "use <string>")
  val sense = new CmdOption("sense", 2, "INT", "use <int> ")
  val threshold = new CmdOption("threshold", 0, "INT", "use <int>")
  val take_sense = new CmdOption("use-sense", 1, "INT", "use <int> ")
}

object scws {

  var vocab = Array[String]()
  var weights = Array[Array[DenseTensor1]]()
  var D = 0
  var V = 0
  var S = 0
  var take_sense = false
  var threshold = 0

  def main(args: Array[String]) {

    /* load the options */
    val opts = new SCWSOpts()
    opts.parse(args)
    val embeddingsFile = opts.embeddingFile.value
    S = opts.sense.value.toInt
    val wordsimFile = opts.wordSimFile.value
    threshold = opts.threshold.value.toInt
    take_sense = if (opts.take_sense.value.toInt == 1) true else false

    println(embeddingsFile)
    // load the embeddings and vocab
    loadVocabAndEmbeddings(embeddingsFile)
    println(wordsimFile)
    doSimTask(wordsimFile)

  }
  def doSimTask(filename: String): Unit = {
    var lineItr = io.Source.fromFile(filename).getLines()
    var not_present = 0
    while (lineItr.hasNext) {
      val details   = lineItr.next.stripLineEnd.split(',') // <word1><comma><word2><comma><score>
      var context = lineItr.next.stripLineEnd.split(' ')
      val w1context = context.map(word => word.toUpperCase).map(word => getID(word)).filter(id => id != -1)
      context = lineItr.next.stripLineEnd.split(' ')
      val w2context = context.map(word => word.toUpperCase).map(word => getID(word)).filter(id => id != -1)
      val w1 = details(0).toUpperCase
      val w2 = details(1).toUpperCase
      val score_org = details(2).toDouble
      val w1id = getID(w1) // get word id
      val w2id = getID(w2) // get word id
      var w1sense = 0
      var w2sense = 0
      if (take_sense && w1id != -1 && w2id != -1) { // use the contexts only take_sense = 1
        w1sense = getSense(w1id, w1context)
        w2sense = getSense(w2id, w2context)
      }
      val score_pred = if (w1id != -1 && w2id != -1) weights(w1id)(w1sense).dot(weights(w2id)(w2sense)).abs * 10 else -1
      if (score_pred != -1) println("%s,%s,%f,%f".format(w1, w2, score_pred, score_org))
      else not_present += 1
    }
    println("total not present : " + not_present)
  }
  private def getSense(word: Int, contexts: Seq[Int]): Int = {
    val contextEmbedding = new DenseTensor1(D, 0)
    (0 until contexts.size).foreach(i => contextEmbedding.+=(weights(contexts(i))(0))) // global context
    var correct_sense = 0
    var max_score = Double.MinValue
    for (s <- 1 until S + 1) {
      val score = contextEmbedding.dot(weights(word)(s)) // find the local context
      if (score > max_score) {
        correct_sense = s
        max_score = score
      }
    }
    correct_sense
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
    weights = Array.ofDim[DenseTensor1](V, S + 1)
    for (v <- 0 until V) {
      val line = lineItr.next.stripLineEnd.split(' ')
      vocab(v) = line(0).toUpperCase
      for (s <- 0 until S + 1) {
        val fields = lineItr.next.stripLineEnd.split("\\s+")
        weights(v)(s) = new DenseTensor1(fields.map(_.toDouble))
        weights(v)(s) /= weights(v)(s).twoNorm
      }
      //print(v + " ")
    }
    println("loaded vocab and their embeddings")
  }
  private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equals(word))
      return i
    return -1
  }
}