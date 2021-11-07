(ns linalg.neanderthal
  (:require
   [nextjournal.clerk :as clerk]
   [uncomplicate.neanderthal.core :as unc :refer (axpy mm scal mv)]
   [uncomplicate.neanderthal.native :as unn :refer (dv dge)]
   [uncomplicate.neanderthal.random :as unr]
   [uncomplicate.neanderthal.math :as unm :refer (cos sin)]))

(def cc clerk/code)
(defn ->vec [x] (into [] x))
(defn ->vvec [x] (mapv ->vec x))

(defn ->plotly [v & ws]
  {:data
   (into
    [{:x [0 (first v)] :y [0 (second v)]
      :name :original}]
    (map #(-> {:x [0 (first %)] :y [0 (second %)]}))
    ws)
   :layout {:width 820
            :height 600
            :legend {:orientation :h}}})

(def x (dv 1 2 3))
(defn rand-vec [n]
  (let [x (unr/rand-uniform! (dv n))]
    (unc/scal (/ 1 (unc/nrm2 x)) x)))


(def z (unr/rand-uniform! (dge 3 2)))

(defn rotate [d]
  (dge 2 2 [(cos d) (sin d) (- (sin d)) (cos d)]))

(defn scale [alpha]
  (scal alpha (dge 2 2 [1 0 0 1])))

(def T (dge 2 2 [1 2 0 2]))
(def U (dge 2 2 [1 0 1 1]))


(comment
  (scale 100)
  (axpy x x)
  (axpy 1 x 1 x)
  (scal -1 x))


(clerk/plotly
 (let [v (rand-vec 2)
       w (unc/mv (rotate (/ Math/PI 2)) v)]
   (->plotly v w)))

(clerk/plotly
 (let [_ 3 ;; otherwise not rendered
       v (rand-vec 2)
       w (unc/mv (rotate (/ Math/PI 2)) v)]
   (->plotly v w)))

(clerk/plotly
 (let [_ 2
       v (rand-vec 2)
       w (unc/mv (rotate Math/PI) v)]
   (->plotly v w)))

(clerk/plotly
 (let [_ 1
       v (rand-vec 2)
       w (unc/mv (rotate (/ Math/PI 3)) v)]
   (->plotly v w)))

(clerk/plotly
 (let [_ 4
       v (rand-vec 2)]
   (->plotly
    v
    (unc/mv (mm U T) v)
    (unc/mv (mm T U) v))))

(cc (unc/mmt z))
(cc (unc/view-ge z 2))
(cc (unc/view-ge (unc/mmt z) 3))

(comment
  (clerk/show!)
  )
