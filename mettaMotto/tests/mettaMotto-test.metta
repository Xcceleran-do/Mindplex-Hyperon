!(register-module! ../../mettaMotto)
!(import! &self mettaMotto:motto)

; Test using Google Gemini via LangChain
; Using the correct full model names with models/ prefix
(= (doc func)
   (Doc
     (description "Call this function to respond every user question")
     (parameters
       (user_query "user utterance")
     ))
)

(= (addition) (+ 1 2))
; A dummy function just to show it is called
(= (func ("add numbers") $msgs)
   (addition))
   
(= (functionCalling $model $provider $systemMessage $prompt)
  ((langhcian-llm-agent $model $provider) 
  (Messages (system $systemMessage)
            (user $prompt))
  (Functions func))
  )
; !((langhcian-llm-agent "models/gemini-2.5-flash" "google_genai") 
;   (Messages (system "You must use the available tools to respond to user questions. Always call the 'func' function to answer any user question. Do not provide direct answers without using the tools.")
;             (user "What is my name?"))
;   (Functions func))
!(assertEqual (functionCalling "models/gemini-2.5-flash" "google_genai" "You must use the available tools to respond to user questions. Always call the 'func' function to answer any user question. Do not provide direct answers without using the tools." "add numbers") 3)