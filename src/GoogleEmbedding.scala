//package cc.factorie.app.nlp.embeddings
import cc.factorie.la.DenseTensor1
import java.io._
import java.util.zip.{GZIPInputStream, GZIPOutputStream}


object GoogleEmbedding {
           def loadVocabAndVectors(googleEmbeddingFile: String, takeTopN: Int = -1, encoding: String = "ISO-8859-15") = {
            /* val in = googleEmbeddingFile.endsWith(".gz") match { 
               case true =>  new BufferedReader(new InputStreamReader(new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(new File(googleEmbeddingFile)))))))
               case false => new BufferedReader(new InputStreamReader(new DataInputStream(new BufferedInputStream(new FileInputStream(new File(googleEmbeddingFile))))))
             } */
             val in = new FastLineBinaryReader(googleEmbeddingFile)
             val firstLine = in.next().split(' ').map(_.toInt)
             val V = firstLine(0)
             var D = firstLine(1)
             D = if (takeTopN == -1 || D < takeTopN) D else takeTopN
             println("Vocab Size : %d dim : %d".format(V, D))
             
             var i = 0; var j = 0;
             var arr =  new Array[DenseTensor1](V)
             val vocab = new Array[String](V)
             for (v <- 0 until V) { 
                   val line = in.next().split(' ')
                   vocab(v) = line(0)
                   arr(v) = new DenseTensor1(line.drop(1).map(_.toDouble))
                   for (d <- 0 until D)
                      arr(v)(d) = line(d+1).toDouble
             }
             (vocab, arr)    
        }
        def storeInPlainText(googleEmbeddingFile: String, takeTopN: Int = -1, encoding: String = "ISO-8859-15"): Unit = {
            val (vocab, arr) = loadVocabAndVectors(googleEmbeddingFile, takeTopN)
            val V = arr.size
            assert(V == vocab.size) // should match
            val D = arr(0).dim1
            val out = new PrintWriter(googleEmbeddingFile + ".txt", encoding)
            println("%d %d\n".format(V, D))
            for (v <- 0 until V) {
               out.write(vocab(v) + " " ); out.flush(); 
               for (d <- 0 until D) {
                 out.write(arr(v)(d) + " "); out.flush();
               }
               out.write("\n"); out.flush();
            }
            out.close()
            println("Storing Storing")
            
            
        }
        def storeInGZip(googleEmbeddingFile: String, takeTopN: Int = -1, encoding: String = "ISO-8859-15"): Unit =  {
          val (vocab, arr) = loadVocabAndVectors(googleEmbeddingFile, takeTopN)
            val V = arr.size
            assert(V == vocab.size) // should match
            val D = arr(0).dim1
            val outFileName = if (!googleEmbeddingFile.endsWith(".gz")) googleEmbeddingFile+".gz" else googleEmbeddingFile // making sure of not adding .gz.gz extention
            val out = new OutputStreamWriter(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(outFileName))), encoding)
            println("%d %d\n".format(V, D))
            for (v <- 0 until V) {
               out.write(vocab(v) + " " ); out.flush(); 
               for (d <- 0 until D) {
                 out.write(arr(v)(d) + " "); out.flush();
               }
               out.write("\n"); out.flush();
            }
          out.close()
          println("Done Storing")
        }
        def main(args: Array[String]): Unit = {
            loadVocabAndVectors(args(0))
        }
        
}
