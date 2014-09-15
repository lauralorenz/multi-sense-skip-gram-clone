import scala.util.Random

object test {
    def main(args : Array[String]) {
        val rng = new Random()
        val prob = Array( 0.3454330045802492,0.3142832554253032,0.2622475537217671,0.07)
        val count = new Array[Int](prob.size)
        for (i <- 0 until count.size) count(i) = 0
        for (i <- 0 until 1e5.toInt) {
          val x = cc.factorie.maths.nextDiscrete(prob)(rng)
          count(x) += 1
        }
        var sum = 0.0; count.foreach(y => sum += y)
        count.foreach(y => println(y/sum))
    }
}