(ns linalg.intro
  (:require
   [nextjournal.clerk :as clerk]
   [uncomplicate.neanderthal.core :as unc]
   [uncomplicate.neanderthal.native :as unn]))

;; # Strategy
;; Going from example first, then abstract.

;; # Why caring about Linear Algebra?

;; The area where linear algebra is applied the most are:

;; - Statistics and Machine Learning;
;; - Solving a system of equations (optimization, budgeting);
;; - Computer vision and graphics;
;; - Simulations of stochastic systems [markov chains, graph algorithms].

;; The reason linear algebra is necessary for these fields is thanks to the
;; concept of *single instruction mutiple data* (`SIMD`), which is roughly
;; applying the same function to multiple data point.

(def plus #(apply mapv + %&))
(defn times [alpha x] (mapv (partial * alpha) x))
(defn dot [x y] (reduce + (mapv * x y)))
;; Suppose matrix are row majors
(defn transpose [Y] (apply mapv vector Y))
(defn %*% [X Y]
  (mapv (fn [x] (mapv #(dot x %) (transpose Y))) X))

(plus [1 2] [0 1])
(times 3 [1 2])
(dot [1 2] [0 1])

(%*% [[1 0] [0 1]]
     [[1 2] [3 4]])

(%*% [[0 1] [1 0]]
     [[1 2] [3 4]])

(%*% [[-1 0] [0 1]]
     [[1 2] [3 4]])

;; ## Speed and Algorithms

;; The reason linear algebra is a necessity in practice for these fields is thanks
;; to the concept and the hardware implementation of *single instruction multiple
;; data* (`SIMD`), in other words the art of applying the same operation to
;; multiple data point. Basically, CPU and GPU have specialized instructions to
;; perform linear algebra operations which speeds up operations by magnitude of
;; orders (depending on the size of the problem).

;; The gain speed allowed to create novel algorithms (such as the bootstrap
;; algorithm in statistics).

;; Depending on the structure of your problem, you could leverage the shape of
;; your matrices to speed up even more the computations.


;; # Linear Algebra Concepts

;; ## Vector Spaces

;; The most used vector space is probably the cartesian product of $\mathbb{R}$,
;; that is $\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$, and generally
;; $\mathbb{R}^n$.



;; Other more abstract vector spaces: the space of function from $\mathbb{R}$ to
;; $\mathbb{R}$.

;; ## Linear Mappings

;; Example of complex linear mapping: the derivative of a function is a linear
;; operation, as
;;$T(\alpha f+ \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta  g' = \alpha T(f) + \beta T(g)$.
