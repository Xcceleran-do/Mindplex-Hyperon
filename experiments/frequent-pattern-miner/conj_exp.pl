:- use_module(library(lists)).
:- use_module(library(pairs)).

% unique_combinations_star(+Exprs, +Size, -Results)
% Generates conjunctions of size Size where all clauses share exactly one hub
% variable and no other variable is shared across any pair of clauses.
unique_combinations_star(Exprs, Size, Results) :-
    parse_k(Size, K),
    ( K =< 0
    -> Results = []
    ; length(Exprs, N),
      K > N
    -> Results = []
    ; build_infos(Exprs, Infos),
      collect_hubs(Infos, Hubs),
      findall(SortedCombo,
              ( member(Hub, Hubs),
                pool_with_hub(Infos, Hub, Pool),
                length(Pool, PoolLen),
                PoolLen >= K,
                combos_for_hub(Pool, Hub, K, Combo),
                sort(Combo, SortedCombo)
              ),
              RawCombos),
      sort(RawCombos, UniqueCombos),
            maplist(normalize_combo_vars, UniqueCombos, NormalizedCombos),
            maplist(wrap_conjunct, NormalizedCombos, WrappedCombos),
            include(conjunct_has_engagement, WrappedCombos, Results)
    ).

parse_k(Size, K) :-
    ( integer(Size) -> K = Size
    ; number(Size) -> K is floor(Size)
    ; atom(Size) -> ( atom_number(Size, K) -> true ; K = 0 )
    ; string(Size) -> ( number_string(K, Size) -> true ; K = 0 )
    ; K = 0
    ).

% cut-first-char(+Input, -Output)
% Drops the first character from an atom/string like l$x -> $x.
'cut-first-char'(Input, Output) :-
    ( var(Input)
    -> Output = Input
    ; atom(Input)
    -> atom_chars(Input, Chars),
       drop_first_char(Chars, RestChars),
       atom_chars(RestAtom, RestChars),
       Output = RestAtom
    ; string(Input)
    -> string_chars(Input, Chars),
       drop_first_char(Chars, RestChars),
       string_chars(RestString, RestChars),
       Output = RestString
    ; Output = Input
    ).

drop_first_char([_|Rest], Rest) :- !.
drop_first_char([], []).

build_infos([], []).
build_infos([Expr|Rest], [info(Expr, VarsSet, Functor)|Infos]) :-
    extract_var_keys(Expr, VarsSet),
    expr_functor(Expr, Functor),
    build_infos(Rest, Infos).

expr_functor([F|_], Functor) :-
    atom(F),
    !,
    Functor = F.
expr_functor(_, '').

collect_hubs(Infos, Hubs) :-
    findall(V,
            ( member(info(_, Vars, _), Infos),
              member(V, Vars)
            ),
            Vars),
    list_to_set(Vars, Hubs).

pool_with_hub(Infos, Hub, Pool) :-
    include(info_has_hub(Hub), Infos, Pool).

info_has_hub(Hub, info(_, Vars, _)) :-
    memberchk(Hub, Vars).

combos_for_hub(Pool, Hub, K, Combo) :-
    choose_combo(Pool, Hub, K, [], [], Combo).

choose_combo(_, _, 0, SelectedInfos, _, Combo) :-
    findall(Expr, member(info(Expr, _, _), SelectedInfos), Combo).
choose_combo([Info|Rest], Hub, K, SelectedInfos, UsedFunctors, Combo) :-
    K > 0,
    ( can_add_info(Hub, Info, SelectedInfos, UsedFunctors, NewUsedFunctors),
      K1 is K - 1,
      choose_combo(Rest, Hub, K1, [Info|SelectedInfos], NewUsedFunctors, Combo)
    ; choose_combo(Rest, Hub, K, SelectedInfos, UsedFunctors, Combo)
    ).

can_add_info(Hub, info(_, Vars, Functor), SelectedInfos, UsedFunctors, NewUsedFunctors) :-
    functor_ok(Functor, UsedFunctors, NewUsedFunctors),
    compatible_with_selected(Hub, Vars, SelectedInfos).

functor_ok('', UsedFunctors, UsedFunctors) :- !.
functor_ok(Functor, UsedFunctors, [Functor|UsedFunctors]) :-
    atom(Functor),
    \+ memberchk(Functor, UsedFunctors).

compatible_with_selected(_, _, []) :- !.
compatible_with_selected(Hub, Vars, [info(_, Vars2, _)|Rest]) :-
    only_hub_shared(Hub, Vars, Vars2),
    compatible_with_selected(Hub, Vars, Rest).

only_hub_shared(Hub, Vars1, Vars2) :-
    shared_vars(Vars1, Vars2, Inter),
    Inter == [Hub].

shared_vars(Vars1, Vars2, Shared) :-
    findall(V,
            ( member(V, Vars1),
              member(V2, Vars2),
              V == V2
            ),
            Shared0),
    list_to_set(Shared0, Shared).

extract_var_keys(Term, Keys) :-
    extract_var_keys(Term, [], Keys0),
    list_to_set(Keys0, Keys).

extract_var_keys(Var, Acc, [Key|Acc]) :-
    var(Var),
    !,
    term_to_atom(Var, Name),
    atom_concat('$', Name, Key).
extract_var_keys(Atom, Acc, [Atom|Acc]) :-
    atom(Atom),
    atom_chars(Atom, ['$'|_]),
    !.
extract_var_keys(Str, Acc, [StrKey|Acc]) :-
    string(Str),
    string_chars(Str, ['$'|_]),
    !,
    atom_string(StrKey, Str).
extract_var_keys(List, Acc, Keys) :-
    is_list(List),
    !,
    extract_var_keys_list(List, Acc, Keys).
extract_var_keys(Term, Acc, Keys) :-
    compound(Term),
    !,
    Term =.. [_|Args],
    extract_var_keys_list(Args, Acc, Keys).
extract_var_keys(_, Acc, Acc).

extract_var_keys_list([], Acc, Acc).
extract_var_keys_list([H|T], Acc, Keys) :-
    extract_var_keys(H, Acc, Acc1),
    extract_var_keys_list(T, Acc1, Keys).

normalize_combo_vars(Combo, Normalized) :-
    normalize_terms(Combo, [], _, Normalized).

normalize_terms([], Map, Map, []).
normalize_terms([T|Ts], Map0, Map, [N|Ns]) :-
    normalize_term(T, Map0, Map1, N),
    normalize_terms(Ts, Map1, Map, Ns).

normalize_term(Var, Map, Map, Var) :-
    var(Var),
    !.
normalize_term(Atom, Map0, Map, Var) :-
    atom(Atom),
    atom_chars(Atom, ['$'|_]),
    !,
    lookup_or_add_var(Atom, Map0, Map, Var).
normalize_term(Str, Map0, Map, Var) :-
    string(Str),
    string_chars(Str, ['$'|_]),
    !,
    atom_string(Atom, Str),
    lookup_or_add_var(Atom, Map0, Map, Var).
normalize_term(List, Map0, Map, Normalized) :-
    is_list(List),
    !,
    normalize_terms(List, Map0, Map, Normalized).
normalize_term(Term, Map0, Map, Normalized) :-
    compound(Term),
    !,
    Term =.. [F|Args],
    normalize_terms(Args, Map0, Map, NormArgs),
    Normalized =.. [F|NormArgs].
normalize_term(Term, Map, Map, Term).

lookup_or_add_var(Key, Map0, Map, Var) :-
    ( memberchk(Key-Var0, Map0)
    -> Var = Var0,
       Map = Map0
    ; Var = _,
      Map = [Key-Var|Map0]
    ).

wrap_conjunct(Combo, [conjunct, [','|Combo]]).

required_functor_keywords([engagement]).

conjunct_has_engagement([conjunct, [','|Clauses]]) :-
    required_functor_keywords(Keywords),
    member(Clause, Clauses),
    clause_has_required_keyword(Clause, Keywords),
    !.
conjunct_has_engagement(_) :- false.

clause_has_required_keyword([Functor|_], Keywords) :-
    atom(Functor),
    member(Keyword, Keywords),
    sub_atom(Functor, 0, _, _, Keyword).

% promote_engagement_conj(+Conjunction, -Result)
% Move engagement-related clauses to the end of a conjunction.
promote_engagement_conj(Conjunction, Promoted) :-
    ( Conjunction = [','|Clauses]
    -> required_functor_keywords(Keywords),
       partition(has_required_keyword(Keywords), Clauses, Matches, Others),
         append(Others, Matches, Promoted)
    ; Promoted = Conjunction
    ).

has_required_keyword(Keywords, Clause) :-
    clause_has_required_keyword(Clause, Keywords).
