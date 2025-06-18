from vllm import LLM, SamplingParams
import torch
import traceback

prompts_128 = [
    "Describe in vivid detail a bustling marketplace in a fantasy city, focusing on sights, sounds, and smells. ",
    "Explain the complete lifecycle of a star, from nebula to its potential end states, for a curious adult. Please include temperatures at each stage.",
    "Outline a complex plot for a detective novel set in a futuristic, cyberpunk world. Include key twists. ",
    "Imagine Earth if dinosaurs never went extinct. Detail human civilization\'s development in this scenario.",
    "Detail a multi-decade plan for a self-sustaining human colony on Mars. Include all crucial stages.",
    "Compose a comprehensive travel guide for a fictional lost island, detailing its geography and culture.",
    "Explain the intricate socio-political system of a bee hive or ant colony as if it were a nation.",
    "Develop a detailed history of a mythical kingdom, including its founding, golden age, and eventual fall.",
    "Describe the step-by-step process of building a complex medieval trebuchet from scratch. Be thorough.",
    "Explore the philosophical implications of true artificial general intelligence achieving consciousness.",
    "Craft an origin story for a superhero whose powers are based on manipulating probabilities. Be creative.",
    "Explain quantum superposition, entanglement, and uncertainty in depth, using multiple clear analogies.",
    "Design a new, complex board game. Detail its rules, components, and strategic depth for players.",
    "Chronicle the hypothetical first contact scenario between humanity and a peaceful, advanced alien race.",
    "Write a comprehensive biography of a fictional historical inventor who changed their world profoundly.",
    "Detail the ecological succession of a forest after a catastrophic wildfire, over several decades.",
    "Imagine a world where magic is real but strictly regulated by a global council. Detail daily life.",
    "Explain the entire process of winemaking, from grape cultivation to bottling and aging, in detail.",
    "Compare and contrast core tenets of Stoicism & Epicureanism, with examples of their application.",
    "Design an alien ecosystem on a planet with two suns, detailing its unique flora and fauna.",
    "Describe the rise and fall of a fictional ancient empire known for its advanced water engineering.",
    "Outline a strategy for reversing climate change, addressing technological and social aspects fully.",
    "Explain plate tectonics theory and its vast impact on Earth\'s geology and life through eons.",
    "Create a detailed character profile for a morally grey anti-hero in a dark fantasy setting.",
    "Describe the intricate political landscape of a galaxy with multiple warring alien federations.",
    "Detail the complete brewing process for a complex craft beer, from mashing to fermentation & beyond.",
    "Explore the societal impact if humans suddenly developed reliable telepathic abilities overnight.",
    "Narrate a water molecule\'s journey through Earth\'s entire water cycle, with intricate details.",
    "Devise a comprehensive plan for exploring and colonizing the ocean depths, including technologies.",
    "Explain 'the sublime' in art and nature, providing diverse examples and thorough analysis.",
    "Construct an alternate history timeline where the Library of Alexandria was never destroyed.",
    "Describe complex cultural traditions of a secluded, ancient mountain-dwelling tribe in detail."
    "Write a story about a character who discovers they can communicate with animals, and the adventures that follow.",
    "Imagine a world where dreams can be recorded and shared. Explore the societal and personal implications.",
    "Describe the journey of a lone astronaut stranded on an unexplored alien planet.",
    "Create a mythological creature and write its origin story, detailing its powers and significance.",
    "Outline a screenplay for a historical epic set during the construction of the Great Wall of China.",
    "Develop a concept for a video game where players must survive in a post-apocalyptic world reclaimed by nature.",
    "Write a short story about a magical library where books come to life.",
    "Imagine a society where emotions are suppressed by technology. What happens when someone starts to feel?",
    "Describe the culture and traditions of a nomadic tribe traveling across a vast desert landscape.",
    "Create a detailed plan for a theme park based on ancient mythology from around the world.",
    "Write a poem about the beauty and power of a thunderstorm.",
    "Imagine a future where humans have colonized other planets in our solar system. Describe life on one of these colonies.",
    "Develop a mystery story where the detective is a sentient robot.",
    "Describe the experience of attending a grand masquerade ball in a Venetian palace.",
    "Write a children's story about a brave little firefly who saves his friends.",
    "Imagine a world where plants are the dominant life form and humans are a minority. How does society function?",
    "Create a recipe for a magical potion and describe its effects in detail.",
    "Outline a documentary about the secret lives of urban wildlife.",
    "Develop a story about a time traveler who accidentally changes a major historical event.",
    "Describe the architecture and atmosphere of a hidden city in the Himalayas.",
    "Write a song about the changing seasons and their impact on nature.",
    "Imagine a future where food is synthesized. What are the culinary experiences like?",
    "Create a character who can control the weather, and explore the responsibilities and challenges they face.",
    "Describe the sights and sounds of a vibrant coral reef ecosystem.",
    "Write a play about a family reunion where long-held secrets are revealed.",
    "Imagine a world where gravity works differently. How does this affect daily life and technology?",
    "Develop a series of riddles that lead to a hidden treasure.",
    "Describe the training and lifestyle of a knight in a medieval fantasy kingdom.",
    "Write a horror story about a haunted house that adapts to its inhabitants\' fears.",
    "Imagine a society that lives on floating islands in the sky. Detail their culture and technology.",
    "Create a new form of martial art and describe its philosophy and techniques.",
    "Outline a travel blog about a journey along the Silk Road in ancient times.",
    "Develop a story about a group of adventurers searching for a legendary lost city.",
    "Describe the process of terraforming a barren planet to make it habitable.",
    "Write a monologue from the perspective of an ancient tree witnessing centuries of change.",
    "Imagine a future where humans can upload their consciousness into a virtual reality. Explore the pros and cons.",
    "Create a set of laws for a utopian society and explain the reasoning behind them.",
    "Describe the sounds and atmosphere of a dense, primeval jungle.",
    "Write a humorous story about a series of mishaps that occur during a camping trip.",
    "Imagine a world where music has magical properties. How is it used and controlled?",
    "Develop a character who is a master artisan, and describe their creative process.",
    "Describe the experience of navigating a vast, labyrinthine underground cave system.",
    "Write a science fiction story about a first encounter with an alien intelligence that communicates through mathematics.",
    "Imagine a society where individuals are assigned their life partners by an AI. What are the consequences?",
    "Create a detailed map of a fictional continent, including its geographical features and major cities.",
    "Outline a historical fiction novel set during the Renaissance, focusing on an artist or inventor.",
    "Develop a story about a chef who can infuse emotions into their cooking.",
    "Describe the challenges and rewards of building a community in a remote, off-grid location.",
    "Write a poem celebrating the diversity of human languages and cultures.",
    "Imagine a future where space travel is commonplace. Describe a luxury cruise through the asteroid belt.",
    "Create a new pantheon of gods and goddesses, detailing their domains and relationships.",
    "Describe the annual migration of a fantastical herd of creatures across a breathtaking landscape.",
    "Write a mystery set in a remote research station in Antarctica.",
    "Imagine a world where people can shapeshift into animals. How does this impact society and individual identity?",
    "Develop a choose-your-own-adventure story set in a haunted forest.",
    "Describe the design and function of a futuristic transportation system.",
    "Write a story about a lonely lighthouse keeper who befriends a mythical sea creature.",
    "Imagine a society built within the boughs of giant, ancient trees.",
    "Create a detailed festival celebrating the harvest in a fantasy agricultural community.",
    "Outline a story about an archaeologist who uncovers an artifact that rewrites history.",
    "Develop a narrative for a ballet based on a classic fairy tale, but with a dark twist.",
    "Describe the inner workings of a clockwork city powered by steam and ingenuity.",
    "Write a comedic play about a group of aliens attempting to understand human customs.",
    "Imagine a world where shadows have a life of their own. Explore the implications.",
    "Create a survival guide for navigating a world overrun by giant insects.",
    "Describe the experience of witnessing a total solar eclipse from a unique vantage point.",
    "Write a short story about a sentient AI that yearns for a physical body.",
    "Imagine a future where fashion incorporates living, bioluminescent materials.",
    "Develop a legend about a hidden oasis in a vast, unforgiving desert.",
    "Describe the training regimen and daily life of an elite space marine.",
    "Write a folk tale explaining the origin of a particular constellation.",
    "Imagine a city that physically rearranges itself every night.",
    "Create a detailed alien language, including its grammar and writing system.",
    "Outline a thriller about a cryptographer who deciphers a message predicting a global catastrophe.",
    "Develop a story about a musician whose melodies can heal or harm.",
    "Describe the atmosphere and attractions of an intergalactic carnival.",
    "Write a historical account of a legendary battle between two mythical armies.",
    "Imagine a world where every person is born with a unique, minor magical ability.",
    "Create a detailed blueprint for a sustainable, eco-friendly utopian village.",
    "Describe the journey of a single drop of rain from a cloud to the ocean.",
    "Write a suspenseful story about a group of explorers trapped in an ancient tomb.",
    "Imagine a society where art is forbidden. How do artists express themselves in secret?",
    "Develop a quest for a legendary artifact said to grant immortality.",
    "Describe the sights, sounds, and emotions of a child's first visit to the ocean.",
    "Write a story about a sentient storm that communicates its moods through weather patterns.",
    "Imagine a future where memories can be bought, sold, and traded. What are the ethical dilemmas?",
    "Create a new sport played in zero gravity and describe its rules and equipment.",
    "Outline a fantasy novel about a rebellion against a tyrannical sorcerer-king.",
    "Develop a character who is a 'dream weaver,' able to enter and shape the dreams of others.",
    "Describe the construction and launch of a generation ship destined for a distant star system.",
    "Write a poem about the resilience of nature in the face of human development.",
    "Imagine a world where historical figures can be temporarily brought back to life for interviews.",
    "Create a fictional secret society with its own rituals, symbols, and hidden agenda.",
    "Describe the experience of flying on the back of a giant, mythical bird.",
    "Write a detective story where the main clue is a piece of music.",
    "Imagine a library that contains every book ever written, and every book that ever will be written.",
    "Develop a story about a group of scientists who discover a portal to a parallel universe.",
]

prompts_64 = prompts_128[:64]
prompts_32 = prompts_128[:32]
prompts_16 = prompts_128[:16]
prompts_8 = prompts_128[:8]
prompts_4 = prompts_128[:4]
prompts_2 = prompts_128[:2]
prompts_1 = [prompts_128[1]]
# prompts_1 = ["An apple a day", "A cat is a dog and a dog is a cat"]
prompts_1 = ["The big horoscope"]
# prompts_1 = ["Hel"]

max_tokens = 200
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=max_tokens)

format_llama = lambda s: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>
{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
if __name__ == "__main__":
    #llm = LLM(model="/nfs/checkpoint-tuning/llama3_hf/Meta-Llama-3.1-8B-Instruct/",
    llm = LLM(model="/nfs/scratch-aa/hat_vllm/dpo",
    #llm = LLM(model="/Models/hat_dpo",
          trust_remote_code=True,
          dtype=torch.bfloat16,
          enforce_eager=True,
          tensor_parallel_size=1,
          gpu_memory_utilization=0.9,
          block_size=256,
          disable_cascade_attn=True,
          max_num_batched_tokens=100,
          max_model_len=20000, # Can be set to 100k on A100
          max_num_seqs=8)
    outputs = llm.generate([format_llama(p) for p in prompts_8], sampling_params)
    #outputs = llm.generate([p for p in prompts_1], sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text}")
        print("\n\n\n\n")
        print(len(generated_text))
