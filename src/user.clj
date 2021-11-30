(ns user
  (:require
   [portal.api :as p]
   [nextjournal.clerk :as clerk]))


(defn init []
  (p/open {:portal.launcher/port 53755})
  (add-tap #'p/submit)
  (clerk/serve! {:verbose false :browse? false :watch-paths ["src" "src/linalg"] :port 7777}))

;; Add portal as a `tap> target`

(comment
  (tap> {:h 3 :a 3})
  (init)
  (clerk/build-static-app!
   {:paths ["src/linalg/bench.clj"
            "src/linalg/intro.clj"
            "src/linalg/neanderthal.clj"]})
  )
