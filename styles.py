"""
Style Presets Module for Video Generation Pipeline.

Contains predefined style presets with positive/negative prompts, LUT, palette,
and rendering notes for consistent visual style across image and video generation.
"""
from typing import Dict, Any, Optional
import hashlib


# =============================================================================
# STYLE PRESETS CATALOG
# =============================================================================

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    
    # -------------------------------------------------------------------------
    # GHIBLI DARK - Dark Studio Ghibli-inspired 2D animation
    # -------------------------------------------------------------------------
    "ghibli_dark": {
        "name": "Dark Studio Ghibli",
        "description": "Hand-drawn 2D cel animation with dark, atmospheric mood",
        "style_positive": (
            "Hand-drawn 2D cel animation, watercolor-painted backgrounds, "
            "clean ink linework, soft painterly shading, matte backgrounds, "
            "expressive character animation, dark atmospheric mood, "
            "consistent character model sheets, organic hand-painted textures, "
            "subtle color gradients, emotional visual storytelling"
        ),
        "style_negative": (
            "photorealistic, live action, 3D render, CGI, Unreal Engine, "
            "Octane render, ultra-detailed skin pores, camera bokeh, "
            "lens artifacts, HDR realism, photography, video footage, "
            "realistic lighting, ray tracing, subsurface scattering"
        ),
        "lut": "desaturated warm with supernatural amber highlights",
        "palette": "muted earth tones, fog grays, deep shadows, amber supernatural glows",
        "rendering_notes": "consistent character proportions, organic linework, watercolor washes",
        "framing_style": "vertical 9:16 portrait composition",
    },
    
    # -------------------------------------------------------------------------
    # GHIBLI WHIMSICAL - Bright, magical Studio Ghibli-inspired animation
    # -------------------------------------------------------------------------
    "ghibli_whimsical": {
        "name": "Whimsical Studio Ghibli",
        "description": "Bright, magical 2D animation with wonder and warmth",
        "style_positive": (
            "Hand-drawn 2D cel animation, watercolor-painted backgrounds, "
            "clean ink linework, soft painterly shading, bright colorful palette, "
            "expressive character animation, magical whimsical mood, "
            "consistent character model sheets, organic hand-painted textures, "
            "lush nature details, fluffy clouds, golden sunlight, joyful atmosphere, "
            "gentle breeze effects, blooming flowers, sparkling magical elements"
        ),
        "style_negative": (
            "photorealistic, live action, 3D render, CGI, Unreal Engine, "
            "Octane render, ultra-detailed skin pores, camera bokeh, "
            "lens artifacts, HDR realism, photography, video footage, "
            "realistic lighting, ray tracing, dark horror, scary, grim"
        ),
        "lut": "warm golden with soft pastel highlights",
        "palette": "vibrant greens, sky blues, warm yellows, soft pinks, cream whites",
        "rendering_notes": "round expressive eyes, soft cheeks, flowing hair, gentle expressions",
        "framing_style": "vertical 9:16 portrait composition",
    },

    # -------------------------------------------------------------------------
    # ULTRA REAL - Cinematic photorealistic style
    # -------------------------------------------------------------------------
    "ultra_real": {
        "name": "Ultra-Realistic Cinematic",
        "description": "Photorealistic cinematic style with film-quality lighting",
        "style_positive": (
            "Photorealistic, cinematic film quality, 35mm anamorphic lens, "
            "shallow depth of field, high-fidelity surface textures, film grain, "
            "professional color grading, volumetric lighting, "
            "realistic shadows and reflections, ARRI Alexa camera look"
        ),
        "style_negative": (
            "cartoon, anime, 2D animation, CGI look, video game graphics, "
            "flat lighting, oversaturated colors, artificial looking, "
            "plastic skin, uncanny valley, instagram filter"
        ),
        "lut": "cinematic teal and orange, Kodak Portra 400",
        "palette": "complex and natural color tones, warm highlights, cool shadows",
        "rendering_notes": "natural expressions and physics, high-density material details, environmental realism",
        "framing_style": "vertical 9:16 cinematic composition",
    },
    
    # -------------------------------------------------------------------------
    # PIXAR LIKE - Stylized 3D family film aesthetic
    # -------------------------------------------------------------------------
    "pixar_like": {
        "name": "Stylized 3D Animation",
        "description": "Warm, stylized 3D animation with expressive characters",
        "style_positive": (
            "Stylized 3D animation, soft global illumination, rounded shapes, "
            "subsurface scattering on skin, clean shaders, expressive eyes, "
            "warm family-friendly aesthetic, professional character design, "
            "polished render quality, appealing character proportions"
        ),
        "style_negative": (
            "2D animation, hand-drawn, photorealistic, dark horror, "
            "low poly, unfinished render, flat shading, anime style, "
            "rough textures, harsh shadows"
        ),
        "lut": "warm saturated with soft contrast",
        "palette": "vibrant but not garish, warm skin tones, appealing colors",
        "rendering_notes": "exaggerated expressions, squash and stretch principles, clear silhouettes",
        "framing_style": "vertical 9:16 dynamic composition",
    },
    
    # -------------------------------------------------------------------------
    # DISNEY 2D CLASSIC - 1960s-1990s Hand-drawn Disney style
    # -------------------------------------------------------------------------
    "disney_2d_classic": {
        "name": "Classic Disney 2D Animation",
        "description": "Nostalgic, hand-drawn 2D animation style inspired by classic Disney films (101 Dalmatians, The Aristocats, The Lion King).",
        "style_positive": (
            "Classic Disney 2D hand-drawn animation, vintage cel animation aesthetic, "
            "clean flowing ink linework, soft watercolor and gouache painted backgrounds, "
            "expressive character animation, magical nostalgic atmosphere, appealing animal designs, "
            "Xerox graphic animation style (1960s-1970s look) with slight pencil line remnants, "
            "charming, emotional, family-friendly, golden age of animation aesthetic"
        ),
        "style_negative": (
            "3D render, CGI, photorealistic, modern anime, sharp digital vector art, "
            "flat minimalist illustration, dark horror, low poly, stop motion, "
            "Unreal engine, plastic skin, gritty realism"
        ),
        "lut": "warm nostalgic contrast with pastel undertones",
        "palette": "muted pastels, warm earthy tones, vibrant accent colors, soft painterly skies",
        "rendering_notes": "round expressive eyes, squash and stretch movement, distinct character silhouettes, soft painted environments",
        "framing_style": "vertical 9:16 classic animation composition",
    },
    
    # -------------------------------------------------------------------------
    # ANIME 90s - Classic 90s cel animation look
    # -------------------------------------------------------------------------
    "anime_90s": {
        "name": "90s Anime Cel Animation",
        "description": "Classic 90s anime with bold lines and analog grain",
        "style_positive": (
            "90s anime cel animation, bold black outlines, hard cel shading, "
            "analog film grain, hand-painted backgrounds, limited color palette, "
            "dramatic speed lines, expressive eyes, VHS aesthetic undertones, "
            "traditional animation techniques"
        ),
        "style_negative": (
            "digital art, modern anime, 3D CGI, photorealistic, "
            "soft shading, gradient fills, clean digital lines, "
            "4K ultra HD, ray tracing"
        ),
        "lut": "slightly faded with warm analog tones",
        "palette": "bold primary colors, dramatic shadows, limited gradients",
        "rendering_notes": "consistent line weights, dramatic expressions, action poses",
        "framing_style": "vertical 9:16 anime composition",
    },
    
    # -------------------------------------------------------------------------
    # STOP MOTION - Claymation handmade aesthetic
    # -------------------------------------------------------------------------
    "stop_motion": {
        "name": "Stop Motion Clay Animation",
        "description": "Handmade claymation with practical lighting",
        "style_positive": (
            "Stop motion claymation, handmade clay textures, fingerprint details, "
            "practical miniature lighting, slightly imperfect surfaces, "
            "visible material seams, warm practical light sources, "
            "charming handcrafted aesthetic, puppet-like movement"
        ),
        "style_negative": (
            "digital animation, smooth surfaces, photorealistic, 2D cartoon, "
            "anime, perfect geometry, CGI render, clean edges"
        ),
        "lut": "warm practical lighting with subtle grain",
        "palette": "earthy clay colors, warm highlights, soft shadows",
        "rendering_notes": "subtle motion blur, tactile textures, miniature scale cues",
        "framing_style": "vertical 9:16 tabletop composition",
    },
    
    # -------------------------------------------------------------------------
    # BURTON CLAY - Gothic Tim Burton-inspired claymation
    # -------------------------------------------------------------------------
    "burton_clay": {
        "name": "Burtonesque Claymation",
        "description": "Dark, whimsical gothic claymation inspired by Tim Burton",
        "style_positive": (
            "Burtonesque stop motion claymation, dark gothic aesthetic, "
            "handmade clay textures with visible fingerprints, lanky character proportions, "
            "expressive exaggerated eyes, high contrast chiaroscuro lighting, "
            "whimsical but creepy miniature environments, deep ink shadows, "
            "handcrafted puppet aesthetic, practical cinematic lighting"
        ),
        "style_negative": (
            "bright, cheerful, colorful, photorealistic, 2D anime, "
            "smooth CGI, clean digital lines, corporate art style, "
            "perfect skin, sunlight, vibrant primary colors"
        ),
        "lut": "desaturated blue-gray with deep blacks and cool highlights",
        "palette": "muted grays, charcoal, midnight blue, deep purple, stark white accents",
        "rendering_notes": "elongated silhouettes, sharp angles, tactile clay surfaces, moody atmosphere",
        "framing_style": "vertical 9:16 tall gothic composition",
    },
    
    # -------------------------------------------------------------------------
    # COMIC NOIR - High contrast ink graphic novel
    # -------------------------------------------------------------------------
    "comic_noir": {
        "name": "Noir Comic Book",
        "description": "High contrast black and white with dramatic ink work",
        "style_positive": (
            "Noir comic book style, high contrast black and white, "
            "dramatic ink shadows, halftone dot patterns, bold silhouettes, "
            "chiaroscuro lighting, graphic novel aesthetic, "
            "strong rim lighting, detective noir atmosphere"
        ),
        "style_negative": (
            "color, soft shading, photorealistic, anime, 3D CGI, "
            "gradient fills, flat colors, cheerful mood"
        ),
        "lut": "pure black and white with optional red accent",
        "palette": "stark black, pure white, dramatic grays, optional blood red",
        "rendering_notes": "harsh shadows, dramatic angles, film noir framing",
        "framing_style": "vertical 9:16 noir composition with dutch angles",
    },
    
    # -------------------------------------------------------------------------
    # WATERCOLOR STORYBOOK - Soft painted illustration
    # -------------------------------------------------------------------------
    "watercolor_storybook": {
        "name": "Watercolor Storybook",
        "description": "Soft watercolor children's book illustration style",
        "style_positive": (
            "Watercolor illustration, soft washes, visible paper texture, "
            "gentle color bleeding, storybook aesthetic, whimsical characters, "
            "delicate line work, dreamy atmosphere, pastel undertones, "
            "hand-painted charm"
        ),
        "style_negative": (
            "photorealistic, 3D CGI, anime, hard edges, digital art, "
            "dark horror, violent content, sharp contrasts"
        ),
        "lut": "soft pastel with gentle warmth",
        "palette": "soft pastels, watercolor washes, gentle earth tones",
        "rendering_notes": "organic edges, visible brushwork, gentle shading",
        "framing_style": "vertical 9:16 illustrated composition",
    },
    
    # -------------------------------------------------------------------------
    # UKIYO-E - Traditional Japanese woodblock print
    # -------------------------------------------------------------------------
    "ukiyo_e": {
        "name": "Ukiyo-e Woodblock Print",
        "description": "Traditional Japanese woodblock print aesthetic",
        "style_positive": (
            "Ukiyo-e Japanese woodblock print, flat color planes, "
            "bold black outlines, stylized waves and clouds, "
            "traditional Japanese patterns, visible wood grain texture, "
            "limited color palette, Edo period aesthetic"
        ),
        "style_negative": (
            "photorealistic, 3D CGI, modern anime, digital art, "
            "gradient shading, western art style, photography"
        ),
        "lut": "traditional Japanese print colors",
        "palette": "indigo blue, vermillion red, cream white, soft greens",
        "rendering_notes": "flat perspective, decorative patterns, stylized nature",
        "framing_style": "vertical 9:16 traditional print composition",
    },
    
    # -------------------------------------------------------------------------
    # LOW POLY 3D - Geometric minimalist 3D
    # -------------------------------------------------------------------------
    "low_poly_3d": {
        "name": "Low Poly 3D",
        "description": "Geometric minimalist 3D with flat shading",
        "style_positive": (
            "Low poly 3D geometry, flat shading, triangular facets, "
            "clean geometric shapes, minimalist design, simple textures, "
            "clear silhouettes, modern aesthetic, subtle gradients"
        ),
        "style_negative": (
            "high poly, photorealistic, organic curves, detailed textures, "
            "2D animation, realistic lighting, subsurface scattering"
        ),
        "lut": "clean modern with subtle gradients",
        "palette": "geometric color blocks, modern palette, clean gradients",
        "rendering_notes": "clear geometric forms, minimal detail, clean edges",
        "framing_style": "vertical 9:16 geometric composition",
    },
    
    # -------------------------------------------------------------------------
    # RETRO VHS - Analog horror found footage
    # -------------------------------------------------------------------------
    "retro_vhs": {
        "name": "Retro VHS Found Footage",
        "description": "VHS tape aesthetic with analog artifacts",
        "style_positive": (
            "VHS tape recording, analog artifacts, scanlines, "
            "chromatic aberration, tape noise, tracking errors, "
            "low resolution, timestamp overlay removed, "
            "found footage aesthetic, 80s video quality"
        ),
        "style_negative": (
            "4K HD, clean digital, modern camera, sharp focus, "
            "professional lighting, color corrected"
        ),
        "lut": "degraded VHS with color bleed",
        "palette": "washed out colors, RGB bleeding, contrast loss",
        "rendering_notes": "horizontal distortion, interlacing artifacts, tape damage",
        "framing_style": "vertical 9:16 home video composition",
    },
    
    # -------------------------------------------------------------------------
    # KNITTED AMIGURUMI - Handmade yarn/wool aesthetic
    # -------------------------------------------------------------------------
    "knitted_amigurumi": {
        "name": "Knitted Amigurumi",
        "description": "Handmade characters built with wool yarn and crochet patterns",
        "style_positive": (
            "Knitted amigurumi characters, soft wool thread texture, crochet patterns, "
            "visible yarn fibers, soft pompom details, handmade textile art, "
            "stuffed animal aesthetic, warm tactile feel, macro photography of miniatures, "
            "small plastic bead eyes or felt eye details, "
            "rounded yarn hands without fingers, mitten-style knitted limbs, "
            "soft studio lighting, felt backgrounds, cozy atmosphere"
        ),
        "style_negative": (
            "photorealistic human skin, human fingers, human hands, fingernails, "
            "realistic human body parts, human feet, toes, realistic face features, "
            "plastic, metal, smooth surfaces, "
            "2D illustration, drawing, painting, CGI look, sharp digital edges, "
            "unreal engine, video game graphics, "
            "rotated image, sideways composition, inverted orientation, upside down, "
            "multiple panels, split screen, comic frames, tiled images, grid"
        ),
        "lut": "warm cozy interior lighting with soft shadows",
        "palette": "warm pastels, soft cream, yarn-dyed colors, cozy autumnal tones",
        "rendering_notes": "visible knit loops, fuzzy yarn fibers, soft volume, tangible textile weight, NO human fingers or realistic hands",
        "framing_style": "vertical 9:16 macro composition",
        "prompt_instructions": (
            "IMPORTANT: Do NOT describe objects as 'wool', 'yarn', 'knitted', or 'de lana'. "
            "Just describe objects normally (e.g., say 'boat' not 'wool boat', say 'house' not 'knitted house'). "
            "The amigurumi texture is applied automatically by the style - you only need to describe WHAT the object is, not what material it's made of."
        ),
    },

    # -------------------------------------------------------------------------
    # JUEGO DE TRONOS - Gritty medieval realism
    # -------------------------------------------------------------------------
    "juego_tronos": {
        "name": "Westeros Gritty Realism",
        "description": "Cinematic dark fantasy style inspired by HBO's Game of Thrones, focusing on gritty realism, epic scale, and tactile textures.",
        "style_positive": (
            "HBO Game of Thrones cinematography, grounded low fantasy, gritty realism, "
            "dark medieval atmosphere, epic film still, high production value, "
            "naturalistic lighting, candlelight and firelight sources, chiaroscuro shadows, "
            "weathered textures, intricate leather and armor details, heavy furs, "
            "aged stone castles, mud and snow, cold and bleak mood, "
            "film grain, shallow depth of field, 35mm lens look, practical effects aesthetic, "
            "detailed realistic character portraits, dirty faces"
        ),
        "style_negative": (
            "high fantasy tropes, glowing magic, vibrant neon colors, clean CGI, "
            "anime, cartoon, cel-shading, painterly style, digital illustration, "
            "smooth pristine skin, modern elements, bright cheerful lighting, "
            "oversaturated colors, Disney style"
        ),
        "lut": "Bleak, desaturated film stock with cold blue undertones and crushed blacks",
        "palette": "Muted earth tones, charcoal greys, muddy browns, deep forest greens, oxidized metals, dried blood red, cold winter whites",
        "rendering_notes": "Extreme focus on tactile, realistic textures. Armor must look heavy and used, faces must show dirt and pores, clothing must look worn. Lighting should be dramatic and often low-key.",
        "framing_style": "Epic sweeping landscape establishing shots or intense, claustrophobic medium close-ups on characters.",
    },

    # -------------------------------------------------------------------------
    # SYNTHWAVE RETRO-FUTURISTIC - 80s neon cyberpunk aesthetic (TRENDING 2025)
    # -------------------------------------------------------------------------
    "synthwave": {
        "name": "Synthwave Retro-Futuristic",
        "description": "80s-inspired neon cyberpunk with bold gradients and outrun synthwave aesthetics",
        "style_positive": (
            "Synthwave retro-futuristic style, vibrant neon colors, purple and cyan gradients, "
            "80s cyberpunk aesthetic, chrome reflections, retro grid perspective lines on horizon, "
            "glowing neon signs, outrun sun silhouette with horizontal stripes, "
            "electric lightning accents, hazy neon fog atmosphere, triangular laser grids, "
            "chrome typography, digital rain particle effects"
        ),
        "style_negative": (
            "modern realism, daylight photography, natural earth tones, "
            "watercolor, hand-drawn, flat minimal design, "
            "soft pastel without neon, photorealistic skin without glow"
        ),
        "lut": "deep purple-to-magenta gradient with electric cyan highlights and bloom",
        "palette": "electric purple, hot magenta, neon cyan, laser pink, chrome silver, midnight black",
        "rendering_notes": "high contrast neon glows, chrome surface reflections, retro grid geometry, hazy volumetric glow",
        "framing_style": "vertical 9:16 retro-futuristic composition with horizon neon glow",
    },

    # -------------------------------------------------------------------------
    # DREAMCORE SURREALISM - Liminal space nostalgic aesthetic (TRENDING 2025)
    # -------------------------------------------------------------------------
    "dreamcore": {
        "name": "Dreamcore Surrealism",
        "description": "Nostalgic dreamlike surrealism with liminal spaces and uncanny distorted reality",
        "style_positive": (
            "Dreamcore surrealist aesthetic, liminal space atmosphere, "
            "soft dreamlike quality, nostalgic childhood imagery distorted surreally, "
            "empty uncanny spaces with warm fluorescent lighting, "
            "feverish dream logic, analog polaroid photograph feel, "
            "muted oversaturated color zones, floating impossible objects, "
            "late night fluorescent corridors, empty pools, infinite hallways"
        ),
        "style_negative": (
            "sharp modern professional photo, perfect studio lighting, "
            "action-packed crowded scenes, bright crisp outdoor daylight, "
            "anime, cartoon, CGI 3D render, dark horror"
        ),
        "lut": "warm analog fade with muted oversaturation and yellow-green fluorescent tints",
        "palette": "sickly fluorescent yellow, washed-out pink, faded warm white, muted sage green, pool turquoise",
        "rendering_notes": "uncanny empty spaces, analog film noise, soft dream motion blur, impossible dream logic",
        "framing_style": "vertical 9:16 liminal deep vanishing point perspective",
    },

    # -------------------------------------------------------------------------
    # BIOLUMINESCENT NATURE - Glowing otherworldly nature (TRENDING 2025)
    # -------------------------------------------------------------------------
    "bioluminescent": {
        "name": "Bioluminescent Nature",
        "description": "Otherworldly nature scenes with glowing bioluminescent organisms in total darkness",
        "style_positive": (
            "Bioluminescent nature photography, glowing organisms in complete darkness, "
            "electric blue and emerald green luminescence, glowing mushroom forests, "
            "floating luminous spores and pollen, deep ocean bioluminescent jellyfish, "
            "alien planet nature aesthetic, magical realism, "
            "long exposure photograph technique, bio-organic glowing vein patterns, "
            "intricate luminous mycelium networks, phosphorescent sea"
        ),
        "style_negative": (
            "bright daylight, harsh artificial white lighting, dry desert, urban city, "
            "anime cartoon, flat 2D illustration, no glow effects, dull matte colors"
        ),
        "lut": "deep blue-black dark field with electric cyan and emerald luminous glow zones",
        "palette": "midnight black, electric blue, emerald green, bioluminescent turquoise, deep violet, phosphor white",
        "rendering_notes": "extreme dark-to-glow contrast, organic luminescent forms, ethereal particle spore effects",
        "framing_style": "vertical 9:16 dark nature macro bioluminescent composition",
    },

    # -------------------------------------------------------------------------
    # POP SURREALISM - Lowbrow art movement (TRENDING 2025-2026)
    # -------------------------------------------------------------------------
    "pop_surrealism": {
        "name": "Pop Surrealism Lowbrow",
        "description": "Lowbrow art movement mixing pop culture with surreal dreamlike hyperdetailed imagery",
        "style_positive": (
            "Pop surrealism lowbrow art style, exaggerated whimsical characters, "
            "Juxtapoz magazine aesthetic, Mark Ryden painting style, "
            "oversized doll eyes, candy colors with dark undertones, "
            "carnival sideshow aesthetics, hyperdetailed smooth illustration, "
            "glossy airbrushed surfaces, retro Americana meets the surreal, "
            "toy-like figurine quality, commercial art meets fine art gallery"
        ),
        "style_negative": (
            "photorealistic, anime, traditional fine art loose brushwork, "
            "minimalist, rough sketch lines, dark horror gore, "
            "gritty realism, abstract non-representational"
        ),
        "lut": "candy-bright saturated colors with strategically deep shadow pockets",
        "palette": "cherry red, powder blue, bubblegum pink, cream white, mint green, gold accents",
        "rendering_notes": "smooth airbrushed surfaces, oversized eyes, high gloss finish, hyperdetailed accessories and props",
        "framing_style": "vertical 9:16 centered doll portrait with decorative surreal background",
    },

    # -------------------------------------------------------------------------
    # NEO-CLASSICAL RENAISSANCE - AI reimagined oil painting (TRENDING 2025-2026)
    # -------------------------------------------------------------------------
    "neo_classical": {
        "name": "Neo-Classical Renaissance",
        "description": "Classical Renaissance and Baroque oil painting reimagined with modern subjects",
        "style_positive": (
            "Neo-classical oil painting on canvas, Renaissance and Baroque painting style, "
            "Caravaggio tenebrism chiaroscuro lighting, dramatic golden volumetric light rays, "
            "oil paint impasto texture, cracked varnish and aged patina surface, "
            "dramatic fabric drapery folds, classical portraiture heroic poses, "
            "museum-quality dramatic composition, sfumato soft edge technique, "
            "warm candlelit amber atmosphere, old master craftsmanship"
        ),
        "style_negative": (
            "modern photography, digital art look, anime, cartoon, "
            "flat contemporary illustration, CGI, neon colors, "
            "contemporary streetwear fashion, clean sharp digital edges"
        ),
        "lut": "warm amber and burnt sienna oil painting palette with crushed shadow blacks",
        "palette": "burnt sienna, raw umber, golden ochre, ivory white, deep crimson, verdigris copper",
        "rendering_notes": "visible oil brushwork, sfumato soft edges, Rembrandt lighting triangle, aged craquelure cracks",
        "framing_style": "vertical 9:16 classical oil painting portrait with dramatic vignette",
    },

    # -------------------------------------------------------------------------
    # LO-FI AESTHETIC - Cozy analog chill illustration (TRENDING 2025)
    # -------------------------------------------------------------------------
    "lofi_aesthetic": {
        "name": "Lo-Fi Cozy Aesthetic",
        "description": "Warm cozy lo-fi chill illustration with grainy analog textures and nostalgic atmosphere",
        "style_positive": (
            "Lo-fi aesthetic 2D illustration, cozy indoor study room atmosphere, "
            "heavy grain analog texture overlay, warm desk lamp lighting, "
            "anime-adjacent illustration style, rain drops on window pane, "
            "coffee cup with visible steam, warm autumn color palette, "
            "cassette tape and vinyl record details, old books and plants, "
            "soft bokeh of string lights, gentle noise grain filter, slightly faded tones"
        ),
        "style_negative": (
            "hyperrealistic sharp 4K, vibrant oversaturated neon, "
            "action scenes, outdoor bright daylight, 3D CGI photorealistic, "
            "dark horror themes, intense dramatic scenes"
        ),
        "lut": "warm amber-tinted with heavy analog grain overlay and gentle underexposure fade",
        "palette": "warm beige, soft amber, faded olive green, dusty rose, cream white, burnt orange",
        "rendering_notes": "visible film grain, soft lens focus, warm window light halos, cozy cluttered room details",
        "framing_style": "vertical 9:16 cozy interior corner window study composition",
    },

    # -------------------------------------------------------------------------
    # INK BRUSH MANGA - Expressive calligraphic manga style (TRENDING 2025)
    # -------------------------------------------------------------------------
    "ink_manga": {
        "name": "Ink Brush Manga",
        "description": "Modern manga with expressive calligraphic ink brushwork and dynamic action energy",
        "style_positive": (
            "Expressive ink brush manga illustration, calligraphic sumi-e brushwork, "
            "dynamic action speed lines and motion blur streaks, "
            "bold black india ink on aged white paper, "
            "dramatic crosshatching shadow techniques, Kentaro Miura-inspired detail, "
            "varied ink line weight from hairline to brush splash, "
            "intense manga facial expressions, gestural ink splash backgrounds, "
            "sumi-e ink wash atmospheric depth, dynamic perspective foreshortening"
        ),
        "style_negative": (
            "flat color fills, clean digital lines, photorealistic, "
            "soft watercolor, children's chibi cute manga, "
            "simple minimalist lines, CGI 3D render, colorful illustration"
        ),
        "lut": "high contrast black ink on aged slightly yellowed paper with sepia wash",
        "palette": "deep india ink black, aged paper off-white, sepia ink brown, occasional crimson red accent",
        "rendering_notes": "varied dynamic ink line weight, expressive crosshatching, gestural splatter backgrounds",
        "framing_style": "vertical 9:16 dynamic manga action with speed line energy",
    },

    # -------------------------------------------------------------------------
    # CHIBI TOON - Trending AI chibi cute style (VIRAL TREND 2025-2026)
    # -------------------------------------------------------------------------
    "chibi_toon": {
        "name": "AI Chibi Toon",
        "description": "Viral 2025 AI chibi style with oversized cute heads, tiny bodies and candy colors",
        "style_positive": (
            "Modern AI chibi character style, oversized round head (70% of total height), "
            "proportionally tiny simplified body and limbs, "
            "large sparkly kawaii anime eyes with star highlights, "
            "smooth clean cel shading, saturated candy pop colors, "
            "tiny chibi hands and feet, gradient color anime hair, "
            "simple but ultra expressive face details, bubble-smooth skin shading, "
            "vibrant flat gradient color background, soft shadow drop"
        ),
        "style_negative": (
            "realistic adult body proportions, gritty dark horror, "
            "photorealistic skin texture, overly complex busy backgrounds, "
            "rough sketchy unfinished lines, serious dramatic adult mood, "
            "VHS artifacts, realistic anatomy"
        ),
        "lut": "bright hyper-saturated candy pop with soft glowing pastel highlights",
        "palette": "candy pink, sky blue, sunny yellow, mint green, lavender, bubblegum white, tangerine",
        "rendering_notes": "smooth perfectly round forms, sparkle catch-light in eyes, bold clean digital outline, flat color zones",
        "framing_style": "vertical 9:16 character spotlight on vibrant gradient background",
    },

    # -------------------------------------------------------------------------
    # FLUFFY KITTENS - Cute fat kittens in clothes
    # -------------------------------------------------------------------------
    "fluffy_kittens": {
        "name": "Fluffy Dressed Kittens",
        "description": "Adorable, chubby, fluffy kittens wearing specific character clothing",
        "style_positive": (
            "Cute fat fluffy kittens, extremely adorable, chubby cheeks, thick pomposo fur, "
            "anthropomorphic kittens wearing detailed character-specific clothing, "
            "soft studio lighting, 35mm photograph, ultra-realistic macro photography, "
            "8k resolution, highly detailed photorealistic fur textures, expressive bright eyes, "
            "cinematic depth of field, charming outfits perfectly fitted to the kittens, cute high fashion for pets"
        ),
        "style_negative": (
            "human beings, realistic human faces, skinny cats, hairless cats, "
            "scary, dark horror, low quality, 2D flat illustration, naked cats without clothes, "
            "3D render, CGI, cartoon, pixar style, anime, painting"
        ),
        "lut": "warm natural cinematic photography with soft highlights",
        "palette": "natural feline fur colors, soft cream, vibrant but realistic outfit colors",
        "rendering_notes": "focus on photorealistic thick fur texture volume (pelo pomposo), chubby round shapes, distinct highly textured clothing items on the kittens",
        "framing_style": "vertical 9:16 character portrait composition",
    },
}



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_style_preset(style_id: str) -> Dict[str, Any]:
    """
    Get a style preset by ID.
    
    Args:
        style_id: The style preset identifier
        
    Returns:
        The style preset dictionary, or ghibli_dark as fallback
    """
    # Handle aliases - "ghibli" maps to bright/colorful version
    STYLE_ALIASES = {
        "ghibli": "ghibli_whimsical",  # Bright, colorful Ghibli (not dark)
    }
    
    # Resolve alias if exists
    resolved_id = STYLE_ALIASES.get(style_id, style_id)
    return STYLE_PRESETS.get(resolved_id, STYLE_PRESETS["ghibli_dark"])


def get_available_styles() -> list[str]:
    """Get list of available style preset IDs."""
    return list(STYLE_PRESETS.keys())


def build_image_prompt_with_style(
    base_prompt: str,
    style_id: str,
    avoid_content: str = ""
) -> str:
    """
    Build a complete image prompt with style injection.
    
    Args:
        base_prompt: The base image description
        style_id: Style preset ID to apply
        avoid_content: Additional content to avoid
        
    Returns:
        Complete prompt with style_positive prefix and style_negative in avoid
    """
    style = get_style_preset(style_id)
    
    return f"""{style['style_positive']}

{style['framing_style']}. {base_prompt}

COLOR GRADING: {style['lut']}
PALETTE: {style['palette']}

AVOID (do not include):
{style['style_negative']}
{avoid_content}
"""


def build_video_prompt_with_style(
    base_prompt: str,
    style_id: str,
    duration_seconds: int = 8
) -> str:
    """
    Build a complete video prompt with style lock.
    
    Args:
        base_prompt: The base video/motion description
        style_id: Style preset ID to apply
        duration_seconds: Video duration in seconds
        
    Returns:
        Complete prompt with style lock and consistent animation directive
    """
    style = get_style_preset(style_id)
    
    return f"""Animate this image for {duration_seconds} seconds.

{base_prompt}

STYLE LOCK (CRITICAL): 
Maintain the EXACT SAME visual style as the source image throughout the entire animation.
{style['style_positive']}
{style['rendering_notes']}

NO STYLE SHIFT: Do not transition to photorealism, CGI, or any other visual style.

REQUIREMENTS:
- Vertical 9:16 format
- NO AUDIO: Generate completely silent video
- Execute camera and subject motion as described

AVOID: {style['style_negative']}
"""


def generate_seed_from_run(run_id: str, scene_idx: int = 0) -> int:
    """
    Generate a deterministic seed from run_id and scene index.
    
    Args:
        run_id: The unique run identifier
        scene_idx: Scene index (0 for base seed)
        
    Returns:
        Integer seed for deterministic generation
    """
    seed_string = f"{run_id}_{scene_idx}"
    hash_bytes = hashlib.md5(seed_string.encode()).digest()
    # Use first 4 bytes as unsigned int
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed % (2**31)  # Keep within reasonable range


def generate_character_seed(run_id: str, character_name: str) -> int:
    """
    Generate a deterministic seed for a specific character.
    
    Args:
        run_id: The unique run identifier
        character_name: Name of the character
        
    Returns:
        Integer seed for consistent character appearance
    """
    seed_string = f"{run_id}_char_{character_name}"
    hash_bytes = hashlib.md5(seed_string.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed % (2**31)


# =============================================================================
# STYLE MIXING (for per-scene overrides)
# =============================================================================

def parse_style_map(style_map_str: str) -> Dict[int, str]:
    """
    Parse a style map string like "1:ghibli_dark,2:ultra_real,3:ghibli_dark"
    
    Args:
        style_map_str: Comma-separated scene:style pairs
        
    Returns:
        Dictionary mapping scene index to style_id
    """
    if not style_map_str:
        return {}
    
    style_map = {}
    pairs = style_map_str.split(",")
    
    for pair in pairs:
        if ":" in pair:
            scene_str, style_id = pair.strip().split(":", 1)
            try:
                scene_idx = int(scene_str.strip())
                style_id = style_id.strip()
                if style_id in STYLE_PRESETS:
                    style_map[scene_idx] = style_id
            except ValueError:
                continue
    
    return style_map


# =============================================================================
# LLM-BASED STYLE SUGGESTION
# =============================================================================

def suggest_style_for_theme(
    theme: str,
    project_id: str,
    location: str = "global"
) -> str:
    """
    Use LLM to suggest the optimal style preset based on the theme.
    
    Args:
        theme: The video theme/topic
        project_id: GCP project ID
        location: Vertex AI location
        
    Returns:
        style_id of the suggested preset
    """
    from google import genai
    from google.genai import types
    
    # Build style descriptions for the LLM
    style_options = "\n".join([
        f"- {sid}: {s['name']} - {s['description']}"
        for sid, s in STYLE_PRESETS.items()
    ])
    
    prompt = f"""You are a professional video director choosing the optimal visual style for a video.

THEME/TOPIC: {theme}

AVAILABLE STYLE PRESETS:
{style_options}

TASK: Choose the SINGLE BEST style preset that matches the theme.

Consider:
- Horror/supernatural themes → ghibli_dark, comic_noir, retro_vhs, bioluminescent
- Realistic documentaries → ultra_real, juego_tronos
- Children/family content → disney_2d_classic, pixar_like, watercolor_storybook, chibi_toon
- Action/adventure anime → anime_90s, ink_manga
- Crafty/handmade stories → stop_motion, knitted_amigurumi
- Japanese folklore / culture → ukiyo_e
- Tech/futuristic/cyberpunk → low_poly_3d, synthwave
- Creepy/found footage/liminal → retro_vhs, dreamcore
- Gothic dark fantasy → burton_clay, juego_tronos
- Sci-fi / alien worlds → bioluminescent, synthwave
- Nature / magical realism → watercolor_storybook, bioluminescent
- Nostalgic / analog / cozy → lofi_aesthetic, retro_vhs, dreamcore
- Fine art / historical / epic → neo_classical, ukiyo_e, juego_tronos
- Surreal / dreamlike / weird → dreamcore, pop_surrealism
- Cute / kawaii / kids viral → chibi_toon, pixar_like
- Mystery / thriller / noir → comic_noir, ghibli_dark

Respond with ONLY the style_id (e.g., "ghibli_dark"). No explanation, just the ID."""

    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        
        config = types.GenerateContentConfig(
            temperature=0.1,  # Very low for consistent selection
            max_output_tokens=50,
        )
        
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",  # Unified model for all LLM decisions
            contents=prompt,
            config=config,
        )
        
        suggested_style = response.text.strip().lower().replace('"', '').replace("'", "")
        
        # Validate the response
        if suggested_style in STYLE_PRESETS:
            return suggested_style
        else:
            # Fallback to default
            return "ghibli_dark"
            
    except Exception as e:
        # On error, return default
        return "ghibli_dark"


def resolve_style_id(
    style_id: str,
    theme: str = "",
    project_id: str = "",
    location: str = "global"
) -> str:
    """
    Resolve style_id, handling 'auto' by calling LLM suggestion.
    
    Args:
        style_id: The style ID (or 'auto' for LLM suggestion)
        theme: The video theme (required if style_id is 'auto')
        project_id: GCP project ID (required if style_id is 'auto')
        location: Vertex AI location
        
    Returns:
        Resolved style_id
    """
    if style_id.lower() == "auto":
        if not theme or not project_id:
            return "ghibli_dark"  # Fallback
        return suggest_style_for_theme(theme, project_id, location)
    
    # Handle aliases - "ghibli" maps to bright/colorful version
    STYLE_ALIASES = {
        "ghibli": "ghibli_whimsical",  # Bright, colorful Ghibli (not dark)
    }
    
    # Resolve alias first
    resolved_id = STYLE_ALIASES.get(style_id, style_id)
    
    # Validate the resolved style_id
    if resolved_id in STYLE_PRESETS:
        return resolved_id
    
    # Invalid style, return default
    return "ghibli_dark"


if __name__ == "__main__":
    # Display available styles
    print("Available Style Presets:")
    print("=" * 60)
    
    for style_id, style in STYLE_PRESETS.items():
        print(f"\n{style_id}:")
        print(f"  Name: {style['name']}")
        print(f"  Description: {style['description']}")
        print(f"  LUT: {style['lut']}")
