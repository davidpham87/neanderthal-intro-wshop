(ns user
  (:require
   [portal.api :as p]
   [nextjournal.clerk :as clerk]))


(defn init []
  (p/open)
  (add-tap #'p/submit)
  (clerk/serve! {:verbose false :browse? true :watch-paths ["src" "src/linalg"]}))

;; Add portal as a `tap> target`


(comment
  (tap> {:h 3 :a 3})
  (init)
  )
