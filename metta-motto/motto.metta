!(import! &self motto)

; Test using Google Gemini via LangChain
; Using the correct full model names with models/ prefix
(= (doc func)
   (Doc
     (description "Call this function to respond every user question")
     (parameters
       (user_query "user utterance")
     ))
)

(= (hi) "sitotaw")
; A dummy function just to show it is called
(= (func ("What is my name?") $msgs)
   (hi))
   
!((langhcian-llm-agent "models/gemini-2.5-flash" "google_genai") 
  (Messages (system "You must use the available tools to respond to user questions. Always call the 'func' function to answer any user question. Do not provide direct answers without using the tools.")
            (user "What is my name?"))
  (Functions func))
