(*  Exercice à rendre **)
(* contrat *)
let rec pgcd a b =
  match a, b with
  | _, 0 -> a
  | 0, _ -> b
  | _, _ when a = b -> a
  | _ when a > b -> pgcd (a - b) b
  | _ -> pgcd a (b - a)



(* tests unitaires *)
let pgcd_test a b =
  let abs x = 
    if x<=0 then -x 
    else x in pgcd (abs a) (abs b)



let%test _ = pgcd_test 3 12 = 3
let%test _ = pgcd_test 15 10 = 5


