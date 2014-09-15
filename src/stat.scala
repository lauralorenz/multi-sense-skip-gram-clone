


object stat {
   
   
   def spearmanRankCoff(X: Seq[Double], Y: Seq[Double]): Double = {
       require(X.size > 0, "first seq is empty")
       require(Y.size > 0, "second seq is empty")
       require(X.size == Y.size, "lengths of seq donot match")
       val n = X.size
       val x = new Array[(Double, Int)](n); (0 until n).foreach(i => x(i) = (X(i), i))
       val y = new Array[(Double, Int)](n); (0 until n).foreach(i => y(i) = (Y(i), i))
       val xSorted = x.sortWith( (a, b) => a._1 <= b._1 ) // sort
       val ySorted = y.sortWith( (a, b) => a._1 <= b._1 ) // sort
       val xRanked = assignRank(xSorted)
       val yRanked = assignRank(ySorted)
       pearsonCoff(xRanked, yRanked) // pearson coff between ranked x and ranked y
      
   }
   def kendallTauARankCoff(x: Seq[Double], y: Seq[Double]): Double = {
        var tau = 0.0
        tau
   }
   def kendallTauBRankCoff(x: Seq[Double], y: Seq[Double]): Double = {
       var tau = 0.0
       tau
   }
   
   def pearsonCoff(X: Seq[Double], Y: Seq[Double]): Double = {
       require(X.size > 0, "first seq is empty")
       require(Y.size > 0, "second seq is empty")
       require(X.size == Y.size, "lengths of seq donot match")
       val EX = sampleMean(X)
       val EY = sampleMean(Y)
       var rho_num : Double = 0.0; var rho_denx: Double = 0.0; var rho_deny: Double = 0.0 ; var i = 0; val n = X.size ; 
       while (i < n) {
           val diff_x = X(i) - EX
           val diff_y = Y(i) - EY
           rho_num  += diff_x * diff_y
           rho_denx += diff_x * diff_x 
           rho_deny += diff_y * diff_y
           i += 1
       }
       var rho_den = math.sqrt( rho_denx * rho_deny) 
       val rho = rho_num / rho_den
       rho
         
   }
   // co-variance between two random variables
   def variance(X : Seq[Double], Y : Seq[Double]): Double = {
       require(X.size > 0, "first seq is empty")
       require(Y.size > 0, "second seq is empty")
       require(X.size == Y.size, "lengths of seq donot match")     
       var i = 0; var n = X.size ; var s = 0.0;
       val EX = sampleMean(X)
       val EY = sampleMean(Y)
       while (i < n) {
             val diff_x = X(i) - EX
             val diff_y = Y(i) - EY
             s += diff_x * diff_y
       }
       s/(1.0 * n)
   }
    
   
   // sample mean of random variable
   def sampleMean(X : Seq[Double]): Double = {
        var s: Double = 0.0; var i = 0; val n = X.size ;
        while (i < n) {
          s += X(i)
          i += 1
        }
        s / (1.0* n)
   }
   // sample mean of random vector
   def sampleMean(X: Seq[Seq[Double]]): Seq[Double] = {
        val nrows = X.size; val ncols = X(0).size;
        val means = new Array[Double](ncols)
        var j = 0;
        while (j < ncols) {
          var i = 0; var s = 0;
          while (i < nrows) {
            means(j) = X(i)(j)
            i += 1
          }
          means(j) /= nrows
          j += 1
        }
        means
   }
   // variance of random variable
   def sampleVariance(X: Seq[Double]): Double = sampleVariance(X, sampleMean(X))
   
   // variance of random vector
   def sampleVariance(X: Seq[Double], mean: Double): Double = {
       val n = X.size; var i = 0;
       var s = 0.0
       while (i < n) {
           val diff = mean - X(i)
           s += (diff*diff)
           i += 1
       }
       s / (1.0*n-1)
   }
   // variance of random vector
   def samplevariance(X: Seq[Seq[Double]]):Array[Array[Double]] = {
         val nrows = X.size; val ncols = X(0).size ; // same # of cols  
         val covMatrix = Array.ofDim[Double](ncols, ncols)
         // compute the means
         val means = sampleMean(X)
         // compute the pairwise co-variance (n)*(n-1)
         // space - O(C*C + C)
         // time - O(C^2 * R ) ;; Every column is visited not more than 3 times 
         var x = 0
         while (x < ncols) {
           var y = x 
           while (y < ncols) {
                val EX = means(x)
                val EY = means(y)
                var j = 0; var s = 0.0;
                while (j < nrows) {
                  val diff_x = X(x)(j) - EX
                  val diff_y = X(y)(j) - EY
                  s += diff_x * diff_y
                  j +=1 
                }
                covMatrix(x)(y) = s/(1.0*(nrows-1)) // divide by (n-1) because it is sample co-variance
                covMatrix(y)(x) = s/(1.0*(nrows-1)) // co-variance matrix is symmertric
                y += 1
           }
           x += 1
         }
         covMatrix
   }
   def sampleStdDev(X : Seq[Double]): Double = sampleStdDev(X, sampleMean(X))
   def sampleStdDev(X : Seq[Double], mean: Double): Double = math.sqrt( sampleVariance(X, mean) )
   
   // SUPPORT FUNCTIONS
  
   // assign rank 
   private def assignRank(x: Seq[(Double, Int)]): Array[Double] = {
       var i = 0; val n = x.size; 
       val rankArr = new Array[Double](n)
       while (i < n) {
           var j = i; var r_num  = i+1; var r_den = 1.0; i += 1;
           while (i < n && x(i)._1 == x(i-1)._1) {  r_num += i+1; r_den += 1; i += 1 }  // handle tie
           val rank = r_num / r_den // r_num : sum of position of duplicates (including the ele itself) ; r_den : # of duplicates (including the ele itself)
           while (j < i) {
             val pos = x(j)._2
             rankArr(pos) = rank
             j += 1
           }
       }
       rankArr
   }
  
   
}