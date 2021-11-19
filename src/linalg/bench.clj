(ns linalg.bench
  (:require
   [criterium.core :refer (quick-bench)]
   [nextjournal.clerk :as clerk :refer (code)]
   [uncomplicate.neanderthal.core :as unc
    :refer (mm mv scal axpy)]
   [uncomplicate.neanderthal.native :as unn :refer (dv dge)]
   [uncomplicate.neanderthal.random :as random]))

(random/rng-state unn/native-float 3)

(defn runif [n]
  (into [] (random/rand-uniform! (dv n))))

(defn bench [n]
  (let [x (runif n)
        y (runif n)]
    (quick-bench (reduce + (map * x y)))
    (quick-bench
     (let [x' (dv x)
           y' (dv y)]
       (unc/dot x' y')))))

;; $n=5$
(code (with-out-str (bench 5)))
;; $n=10$
(code (with-out-str (bench 10)))

;; $n=20$
(code (with-out-str (bench 20)))

;; $n=25$
(code (with-out-str (bench 25)))

;; $n=50$
(code (with-out-str (bench 50)))

;; $n=100$
(code (with-out-str (bench 100)))

;; $n=500$
(code (with-out-str (bench 500)))



(defn bench-scale
  "Benchmark of scaling a vector"
  [n]
  (let [x (runif n)
        alpha (first (runif 1))]
    (quick-bench (mapv #(* alpha %) x))
    (quick-bench
     (let [x' (dv x)]
       (into [] (unc/scal! alpha x'))))))

;; $n=5$
(code (with-out-str (bench-scale 5)))

;; $n=50$
(code (with-out-str (bench-scale 50)))

;; $n=100$
(code (with-out-str (bench-scale 100)))

;; $n=200$
(code (with-out-str (bench-scale 200)))

;; $n=500$
(code (with-out-str (bench-scale 500)))

;; $n=10000$
(code (with-out-str (bench-scale 10000)))

(defn bench-add
  "Benchmark of adding 3 vectors"
  [n]
  (let [x (runif n)
        y (runif n)
        z (runif n)]
    (quick-bench (mapv + x y z))
    (quick-bench
     (let [x' (dv x)
           y' (dv y)
           z' (dv z)]
       (into [] (unc/axpy! 1 x' y' z'))))))

;; $n=5$
(code (with-out-str (bench-add 5)))

;; $n=20$
(code (with-out-str (bench-add 20)))

;; $n=50$
(code (with-out-str (bench-add 50)))

;; $n=200$
(code (with-out-str (bench-add 200)))

;; $n=500$
(code (with-out-str (bench-add 500)))

;; $n=1000$
(code (with-out-str (bench-add 1000)))

;; $n=10,000$
(code (with-out-str (bench-add 10000)))

(defn bench-add-no-io
  "Benchmark of scaling a vector"
  [n]
  (let [x (runif n)
        y (runif n)
        z (runif n)
        x' (dv x)
        y' (dv y)
        z' (dv z)]
    (quick-bench (mapv + x y z))
    (quick-bench (unc/axpy! 1 x' y' z'))))

;; $n=5$
(code (with-out-str (bench-add-no-io 5)))

;; $n=20$
(code (with-out-str (bench-add-no-io 20)))

;; $n=10,000$
(code (with-out-str (bench-add-no-io 10000)))

;; $n=20,000$
(code (with-out-str (bench-add-no-io 20000)))

(code
 (with-out-str
   (let [v (dv 1000) ]
     (quick-bench (last (seq v)))
     (quick-bench (peek (into [] v))))))

(comment
  (clerk/show! "src/linalg/bench.clj")
  (tap> "hello")
  )
