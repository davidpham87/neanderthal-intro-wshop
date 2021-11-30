(ns linalg.neanderthal
  (:require
   [nextjournal.clerk :as clerk]
   [uncomplicate.neanderthal.core :as unc :refer (axpy mm scal mv)]
   [uncomplicate.neanderthal.linalg :as unl]
   [uncomplicate.neanderthal.math :as unm :refer (cos sin)]
   [uncomplicate.neanderthal.native :as unn :refer (dv dge)]
   [uncomplicate.neanderthal.random :as unr]))

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


;; Create a vector
(def x (dv 1 2 3))
(def xs [(dv [1 2 3]) (dv 1 2 3) (dv 3)])

(cc
 "[
#RealBlockVector[double, n:3, offset: 0, stride:1]
  [   1.00    2.00    3.00 ]
#RealBlockVector[double, n:3, offset: 0, stride:1]
  [   1.00    2.00    3.00 ]
#RealBlockVector[double, n:3, offset: 0, stride:1]
  [   0.00    0.00    0.00 ]
]")

;; Create a random vector
(defn rand-vec [n]
  (let [x (unr/rand-uniform! (dv n))]
    (unc/scal (/ 1 (unc/nrm2 x)) x)))

;; Create a random 3x2 matrix
(def z (unr/rand-uniform! (dge 3 2)))

;; Column Major
(def z-2 [(dge 3 2 [1 2 3 4 5 6])
          (dge 3 2 [[1 2 3] [4 5 6]])])

;; Slicing data
(cc
 (let [z (dge 3 2 [1 2 3 4 5 6])]
   {:z z
    :col-0 (unc/col z 0)
    :col-1 (unc/col z 1)
    :cols (vec (unc/cols z))
    :row-0 (unc/row z 0)
    :row-1 (unc/row z 1)
    :rows (vec (unc/rows z))}))


;; Defines a rotation matrix in 2D
(defn rotate [d]
  (dge 2 2 [(cos d) (sin d) (- (sin d)) (cos d)]))

;; Defines a homothety matrix in 2D
(defn scale [alpha]
  (scal alpha (dge 2 2 [1 0 0 1])))

;; ## Some function to manipulate the vectors

;; $x + x$
(cc (axpy x x))
;; Same thing as above
(cc (axpy 1 x 1 x))
;; $-x$
(cc (scal -1 x))

;; ## Addition and scaling of vectors in $V=\mathbb{R}^2$

(clerk/plotly
 (let [_ 1
       alpha 0.5
       x (unn/dv [-2 2])
       ax (unc/ax alpha x)
       y (unn/dv [1 1])
       x+y (unc/xpy x y)]
   {:data [{:x [0 (unc/entry x 0)]
            :y [0 (unc/entry x 1)]
            :name :x}
           {:x [0 (unc/entry y 0)]
            :y [0 (unc/entry y 1)]
            :name :y}
           {:x [(unc/entry y 0) (unc/entry x+y 0)]
            :y [(unc/entry y 1) (unc/entry x+y 1)]
            :name :x'}
           {:x [0 (unc/entry ax 0)]
            :y [0 (unc/entry ax 1)]
            :name :alpha*x}
           {:x [0 (unc/entry x+y 0)]
            :y [0 (unc/entry x+y 1)]
            :name :x+y}]}))

;; # Example of vector subspace

;; A view on the space $V=\mathbb{R}^3$

(clerk/plotly
 (let [_ 1]
   {:data [{:z [0]
            :x [0]
            :y [0]
            :type :mesh3d}]
    :layout {:scene {:aspectmode :cube}
             :xaxis {:range [-2 2]}
             :yaxis {:range [-2 2]}}}))

;; Example of vector subspace of $V=\mathbb{R}^3$

(clerk/plotly
 (let [_ 1
       x [1 0 1]
       y [0 1 1]
       args (unc/trans (unn/dge [x y]))
       grid (unc/trans
             (dge 2 100
                  (for [i (range -5 5) j (range -5 5)]
                    [i j])))
       z (unc/mm grid args)
       z2 (unc/mm grid (unc/trans (unn/dge [[1 1 0] [0 -1 1]])))]
   {:data [{:z (->vec (unc/col z 2))
            :x (->vec (unc/col z 0))
            :y (->vec (unc/col z 1))
            :type :mesh3d}
           {:z (->vec (unc/col z2 2))
            :x (->vec (unc/col z2 0))
            :y (->vec (unc/col z2 1))
            :type :mesh3d}]
    :layout {:scene {:aspectmode :cube}
             :xaxis {:range [-2 2]}
             :yaxis {:range [-2 2]}}}))

;; # Example of Rotation (isometric transformation)

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

;; ## Example of non commutativity of the linear maps
(def T (dge 2 2 [1 2 0 2]))
(def U (dge 2 2 [1 0 1 1]))

(clerk/plotly
 (let [_ 4
       v (rand-vec 2)]
   (->plotly
    v
    (unc/mv (mm U T) v)
    (unc/mv (mm T U) v))))

;; ## Example of eigenvalues and eigen vectors
;; The matrix $T$ is given by
;; $$ T = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} $$
;; This only flip the sign of the second element of the vector With a bit of
;; intuition we see that $e_1 = \begin{bmatrix} 1 & 0 \end{bmatrix}$ and
;; $e_2 = \begin{bmatrix} 0 & 1 \end{bmatrix}$ are eigenvectors with the
;; eigenvalues $1$ and $-1$.

(clerk/plotly
 (let [_ 0
       T (dge 2 2 [1 0 0 -1])
       v (dv [1 1])
       w (unc/mv T v)
       e1 (dv [1 0])
       e2 (dv [0 1])]
   (->plotly v w e1 (unc/mv T e1) e2 (unc/mv T e2))))

;; Here is how you can explore the eigenvalue API

(cc
 (let [_    0
       T    (unc/mm (rotate (/ Math/PI 2)) (dge 2 2 [1 0 0 -1]))
       T'   (unc/copy T)
       v    (dv [1 1])
       w    (unc/mv T v)
       evec (dge 2 2)
       vr   (dge 2 2)
       vl   (dge 2 2)
       _    (unl/ev! T' evec vr vl)]
   {:t T
    :t' T'
    :e evec
    :vr vr
    :vl vl}))

(comment

  (unc/nrm2 (dv [0.70705 0.70705]))
  (dge 2 2 [2 0 0 -3])

  (cc (unc/mmt z))
  (cc (unc/view-ge z 2))
  (cc (unc/view-ge (unc/mmt z) 3))
  (clerk/show! "src/linalg/neanderthal.clj")

  )
