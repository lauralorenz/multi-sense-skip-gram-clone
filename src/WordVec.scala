// package cc.factorie.app.nlp.embeddings
import java.nio.charset.Charset

object WordVec {
  def main(args: Array[String]) {
    val opts = new EmbeddingOpts
    opts.parse(args)
    println("Default Charset of this JVM =" + Charset.defaultCharset());
    println("User Provided Charset for this project=" + opts.encoding.value)
    if (opts.cbow.value == 2) {
      println("Leanring Multiple Embeddings")
      val wordEmbedding = new MultiSenseSkipGramEmbeddingModel(opts)
      val st1 = System.currentTimeMillis()
      wordEmbedding.buildVocab()
      val st = System.currentTimeMillis()
      println("time taken to build vocab : " + (st - st1) / 1000.0)
      wordEmbedding.learnEmbeddings()
      val en = System.currentTimeMillis() - st
      println("time taken to learn embedding : " + en / 1000.0)
      wordEmbedding.store()
     /* if (wordEmbedding.dpmeans == 1)
          wordEmbedding.store_dpmeans()
      else if (wordEmbedding.kmeans == 1)
          wordEmbedding.store()
      else
          wordEmbedding.store_normal() */
    }
    else 
    {
      println("Learning Skip-gram or CBOW Embeddings")
      val wordEmbedding = if (opts.cbow.value == 0) new CBOWNegSamplingEmbeddingModel(opts) else new SkipGramNegSamplingEmbeddingModel(opts)
      val st1 = System.currentTimeMillis()
      wordEmbedding.buildVocab()
      val st = System.currentTimeMillis()
      println("time taken (in seconds) to build vocab : " + (st - st1) / 1000.0)
      wordEmbedding.learnEmbeddings()
      val en = System.currentTimeMillis() - st
      println("time taken (in seconds) to learn embedding : " + en / 1000.0)
      wordEmbedding.store()

    }

  }
}