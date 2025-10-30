import pandas as pd


# ============================================================================
# LEVEL 0 FUNCTIONS (PUBLIC API)
# ============================================================================


def process_sentiment_dataframe(df, categories, output_file=None):
    """
    Process a DataFrame with sentiment expressions and add score columns.
    
    Args:
        df: DataFrame with sentiment expression columns
        categories: List of category names to process
        sentiment_dict: Dictionary with sentiment scores
        output_file: Optional file path to save results
        
    Returns:
        DataFrame with added score columns
    """
    import numpy as np
    sentiment_dict = _load_sentiment_database()
    df_copy = df.copy()
    
    for category in categories:
        if category in df_copy.columns:
            df_copy[f'{category}_score'] = df_copy[category].apply(
                lambda x: np.nan if (isinstance(x, list) and len(x) == 0) else _convert_sentiment_to_score(x, sentiment_dict)
            )
    
    if output_file:
        df_copy.to_csv(output_file, index=False)
        print(f"Sentiment scores saved to {output_file}")
    
    return df_copy

def _convert_sentiment_to_score(sentiment_expressions, sentiment_dict):
    """
    Convert sentiment expressions to numerical scores.
    
    Args:
        sentiment_expressions: String with pipe-separated expressions or list of expressions
        sentiment_dict: Dictionary with sentiment scores
        
    Returns:
        Average sentiment score or NaN if no valid expressions
    """
    if sentiment_expressions is None:
        return float('nan')
    
    if isinstance(sentiment_expressions, list) and len(sentiment_expressions) == 0:
        return float('nan')
    
    expressions = _parse_sentiment_input(sentiment_expressions)
    
    if not expressions:
        return float('nan')
    
    return _calculate_average_score(expressions, sentiment_dict)

def _load_sentiment_database(senticnet_file='data/senticnet.csv'):
    """
    Load SenticNet database into a dictionary for fast lookup.
    
    Args:
        senticnet_file: Path to SenticNet CSV file
        
    Returns:
        Dictionary mapping concept strings to sentiment scores
    """
    sentiment_score_df = pd.read_csv(senticnet_file)
    sentiment_dict = {}
    for _, row in sentiment_score_df.iterrows():
        concept = str(row['CONCEPT']).lower().strip()
        score = float(row['SENTIMENT_SCORE'])
        sentiment_dict[concept] = score
    return sentiment_dict

def calculate_film_averages(df, categories, id_column='id', output_file=None):
    """
    Calculate average sentiment scores per film.
    
    Args:
        df: DataFrame with sentiment scores
        categories: List of category names
        id_column: Column name containing film IDs
        output_file: Optional file path to save results
        
    Returns:
        DataFrame with average scores per film
    """
    score_columns = [f'{cat}_score' for cat in categories if f'{cat}_score' in df.columns]
    
    if not score_columns:
        print("No score columns found. Make sure to process sentiment expressions first.")
        return None
    
    film_scores = df.groupby(id_column)[score_columns].mean().reset_index()
    
    if output_file:
        film_scores.to_csv(output_file, index=False)
        print(f"Film average scores saved to {output_file}")
    
    return film_scores


# ============================================================================
# LEVEL 1 FUNCTIONS (CALLED FROM LEVEL 0)
# ============================================================================

def _parse_sentiment_input(sentiment_expressions):
    """
    Parse sentiment input and convert to list of expressions.
    
    Args:
        sentiment_expressions: String or list of sentiment expressions
        
    Returns:
        List of sentiment expression strings
    """
    if isinstance(sentiment_expressions, str):
        return _parse_string_input(sentiment_expressions)
    elif isinstance(sentiment_expressions, list):
        return _filter_empty_expressions(sentiment_expressions)
    else:
        return []


def _calculate_average_score(expressions, sentiment_dict):
    """
    Calculate average sentiment score from a list of expressions.
    
    Args:
        expressions: List of sentiment expression strings
        sentiment_dict: Dictionary with sentiment scores
        
    Returns:
        Average sentiment score or NaN if no valid scores
    """
    import math
    
    valid_expressions = _filter_empty_expressions(expressions)
    if not valid_expressions:
        return float('nan')
    
    scores = []
    for expr in valid_expressions:
        score = _score_expression(expr, sentiment_dict)
        if not math.isnan(score):
            scores.append(score)
    
    if len(scores) == 0:
        return float('nan')
    
    return sum(scores) / len(scores)


def _score_expression(expression, sentiment_dict):
    """
    Score a single sentiment expression with modifier handling.
    
    Args:
        expression: Sentiment expression string
        sentiment_dict: Dictionary with sentiment scores
        
    Returns:
        Sentiment score or NaN if expression is invalid
    """
    import math
    
    if not expression or expression.strip() == "":
        return float('nan')
    
    negated, intensifier_multiplier, base_word = _parse_expression_structure(expression)
    base_score = _get_base_score(base_word, sentiment_dict)
    
    if math.isnan(base_score):
        return float('nan')
    
    modified_score = _apply_intensifier(base_score, intensifier_multiplier)
    final_score = _apply_negation(modified_score, negated, expression, base_score)
    
    return final_score


# ============================================================================
# LEVEL 2 FUNCTIONS (CALLED FROM LEVEL 1)
# ============================================================================

def __parse_as_list_literal(sentiment_expressions):
    """
    Try to parse string as Python list literal.
    
    Args:
        sentiment_expressions: String representation of a list
        
    Returns:
        Parsed list or None if parsing fails
    """
    import ast
    
    try:
        parsed_list = ast.literal_eval(sentiment_expressions)
        if isinstance(parsed_list, list):
            return parsed_list if len(parsed_list) > 0 else None
        return None
    except (ValueError, SyntaxError):
        return None


def _parse_string_input(sentiment_expressions):
    """
    Parse string input into list of expressions.
    
    Args:
        sentiment_expressions: String containing sentiment expressions
        
    Returns:
        List of sentiment expression strings
    """
    if sentiment_expressions.strip() == "":
        return []
    
    if sentiment_expressions.strip().startswith('[') and sentiment_expressions.strip().endswith(']'):
        parsed = __parse_as_list_literal(sentiment_expressions)
        if parsed is not None:
            return parsed
        return sentiment_expressions.split('|')
    else:
        return sentiment_expressions.split('|')


def _filter_empty_expressions(expressions):
    """
    Filter out empty or whitespace-only expressions.
    
    Args:
        expressions: List of expression strings
        
    Returns:
        List of non-empty expressions
    """
    if expressions == ['']:
        return []
    return [expr.strip() for expr in expressions if expr and expr.strip()]


def _parse_expression_structure(expression):
    """
    Parse expression to extract modifiers and base word.
    
    Args:
        expression: Sentiment expression string
        
    Returns:
        Tuple of (negated, intensifier_multiplier, base_word)
    """
    expression = expression.strip().lower()
    words = expression.split()
    
    if len(words) == 1:
        return __parse_single_word(words[0])
    elif len(words) == 2:
        return __parse_two_words(words[0], words[1])
    else:
        return __parse_multiple_words(words, expression)


def _get_base_score(word, sentiment_dict):
    """
    Get base sentiment score for a word from SenticNet database.
    
    Args:
        word: Word to look up
        sentiment_dict: Dictionary with sentiment scores
        
    Returns:
        Sentiment score or NaN if word not found
    """
    word = word.lower().strip()
    
    if word in sentiment_dict:
        return sentiment_dict[word]
    
    word_with_spaces = word.replace('_', ' ')
    if word_with_spaces in sentiment_dict:
        return sentiment_dict[word_with_spaces]
    
    word_with_underscores = word.replace(' ', '_')
    if word_with_underscores in sentiment_dict:
        return sentiment_dict[word_with_underscores]
    
    return float('nan')


def _apply_intensifier(base_score, intensifier_multiplier):
    """
    Apply intensifier multiplier to base sentiment score.
    
    Args:
        base_score: Base sentiment score
        intensifier_multiplier: Multiplier from intensifier words
        
    Returns:
        Modified sentiment score
    """
    return base_score * intensifier_multiplier


def _apply_negation(modified_score, negated, expression, base_score):
    """
    Apply negation with smart handling for prefix negations.
    
    Args:
        modified_score: Score after applying intensifier
        negated: Whether expression contains negation
        expression: Original expression string
        base_score: Original base sentiment score
        
    Returns:
        Final sentiment score with negation applied
    """
    if not negated:
        return modified_score
    
    if len(expression.strip().split()) == 1 and base_score != 0.0:
        if base_score < 0:
            return modified_score * 1.2
        else:
            return -modified_score * 0.8
    else:
        return -modified_score * 0.8


# ============================================================================
# HELPER FUNCTIONS (LEVEL 2)
# ============================================================================

def __parse_single_word(word):
    """
    Parse single-word expression.
    
    Args:
        word: Single word to parse
        
    Returns:
        Tuple of (negated, intensifier_multiplier, base_word)
    """
    if __has_negative_prefix(word):
        return True, 1.0, word
    return False, 1.0, word


def __parse_two_words(modifier, base_word):
    """
    Parse two-word expression (modifier + word).
    
    Args:
        modifier: First word (potential modifier)
        base_word: Second word (potential base word)
        
    Returns:
        Tuple of (negated, intensifier_multiplier, base_word)
    """
    compound_key = f"{modifier} {base_word}"
    if compound_key in INTENSIFIERS:
        return False, INTENSIFIERS[compound_key], ""
    
    if modifier in NEGATIONS:
        return True, 1.0, base_word
    
    if modifier in INTENSIFIERS:
        return False, INTENSIFIERS[modifier], base_word
    
    if base_word in NEGATIONS:
        return True, 1.0, modifier
    
    return False, 1.0, f"{modifier} {base_word}"


def __parse_multiple_words(words, expression):
    """
    Parse multi-word expression with multiple modifiers.
    
    Args:
        words: List of words in expression
        expression: Original expression string
        
    Returns:
        Tuple of (negated, intensifier_multiplier, base_word)
    """
    negated = False
    intensifier_multiplier = 1.0
    base_words = []
    skip_next = False
    
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        
        if i < len(words) - 1:
            compound = f"{word} {words[i+1]}"
            if compound in INTENSIFIERS:
                intensifier_multiplier *= INTENSIFIERS[compound]
                skip_next = True
                continue
        
        if word in NEGATIONS:
            negated = True
        elif word in INTENSIFIERS:
            intensifier_multiplier *= INTENSIFIERS[word]
        else:
            base_words.append(word)
    
    base_word = ' '.join(base_words) if base_words else expression
    
    if base_words and __has_negative_prefix(base_words[0]) and not negated:
        negated = True
    
    return negated, intensifier_multiplier, base_word


def __has_negative_prefix(word):
    """
    Check if word has a negative prefix.
    
    Args:
        word: Word to check
        
    Returns:
        True if word starts with negative prefix, False otherwise
    """
    negative_prefixes = ['un', 'dis', 'mis', 'non', 'in', 'im', 'il', 'ir', 'anti', 'contra', 'counter', 'mal', 'de', 'a']
    return any(word.startswith(prefix) for prefix in negative_prefixes)


# ============================================================================
# CONSTANTS
# ============================================================================

INTENSIFIERS = {
    # Extreme intensifiers (2.5-3.0+)
    'mind-blowingly': 3.0,
    'earth-shattering': 3.0,
    'breathtakingly': 2.8,
    'jaw-droppingly': 2.8,
    'mind-boggling': 2.8,
    'earth-shaking': 2.7,
    'spine-tingling':  2.7,
    'soul-stirring': 2.7,
    'life-changing': 2.7,
    'game-changing': 2.7,
    'world-class': 2.6,
    'top-notch': 2.6,
    'first-rate': 2.6,
    'mind-numbingly': 2.5,
    'devastatingly': 2.5,
    'monumentally': 2.5,
    
    # Strong intensifiers (1.8-2.4)
    'extremely': 2.0,
    'absolutely': 1.9,
    'utterly': 1.9,
    'exceptionally': 1.9,
    'extraordinarily': 1.9,
    'tremendously': 1.8,
    'incredibly': 1.8,
    'amazingly': 1.8,
    'astonishingly': 1.8,
    'stunningly': 1.8,
    'spectacularly': 1.8,
    'magnificently': 1.8,
    'phenomenally': 1.8,
    'overwhelmingly': 1.8,
    'outstandingly': 1.8,
    'sublimely': 1.8,
    'brilliantly': 1.8,
    'marvelously': 1.7,
    'marvellously': 1.7,
    'wonderfully': 1.7,
    'fantastically': 1.7,
    'superbly': 1.7,
    'ultra': 1.7,
    'thoroughly': 1.7,
    'profoundly': 1.7,
    'immensely': 1.7,
    'enormously': 1.7,
    'vastly': 1.7,
    'intensely': 1.7,
    'perfectly': 1.7,
    'masterfully': 1.7,
    'exquisitely': 1.7,
    'beautifully': 1.7,
    'gloriously': 1.7,
    'splendidly': 1.7,
    'tremendously': 1.6,
    'totally': 1.6,
    'completely': 1.6,
    'entirely': 1.6,
    'wholly': 1.6,
    'remarkably': 1.6,
    'significantly': 1.6,
    'substantially': 1.6,
    'considerably': 1.6,
    'exceedingly': 1.6,
    'severely': 1.6,
    'seriously': 1.6,
    'supremely': 1.6,
    'undeniably': 1.6,
    'unquestionably': 1.6,
    'unbelievably': 1.6,
    'ridiculously': 1.6,
    'insanely': 1.6,
    'wildly': 1.6,
    'madly': 1.6,
    'crazy': 1.6,
    'wicked': 1.6,
    'insanely': 1.6,
    'very': 1.5,
    'hugely': 1.5,
    'massively': 1.5,
    'powerfully': 1.5,
    'strongly': 1.5,
    'distinctly': 1.5,
    'clearly': 1.5,
    'obviously': 1.5,
    'definitely': 1.5,
    'decidedly': 1.5,
    'markedly': 1.5,
    'noticeably': 1.5,
    'dramatically': 1.5,
    'remarkably': 1.5,
    'strikingly': 1.5,
    'undoubtedly': 1.5,
    'undeniably': 1.5,
    'positively': 1.5,
    'downright': 1.5,
    'outright': 1.5,
    
    # Moderate intensifiers (1.1-1.4)
    'super': 1.4,
    'too': 1.4,
    'highly': 1.4,
    'heavily': 1.4,
    'greatly': 1.4,
    'awfully': 1.4,
    'terribly': 1.4,
    'dreadfully': 1.4,
    'frightfully': 1.4,
    'horribly': 1.4,
    'wickedly': 1.4,
    'fiercely': 1.4,
    'intensely': 1.4,
    'aggressively': 1.4,
    'brutally': 1.4,
    'savagely': 1.4,
    'really': 1.3,
    'so': 1.3,
    'particularly': 1.3,
    'especially': 1.3,
    'deeply': 1.3,
    'deep': 1.3,  # Added - similar to "deeply"
    'truly': 1.3,
    'certainly': 1.3,
    'surely': 1.3,
    'indeed': 1.3,
    'genuinely': 1.3,
    'authentically': 1.3,
    'sincerely': 1.3,
    'honestly': 1.3,
    'legitimately': 1.3,
    'properly': 1.3,
    'adequately': 1.3,
    'sufficiently': 1.3,
    'real': 1.3,  # Added - emphasizes authenticity/genuineness
    'quite': 1.2,
    'pretty': 1.2,
    'rather': 1.2,
    'fairly': 1.2,
    'genuinely': 1.2,
    'surprisingly': 1.2,
    'unusually': 1.2,
    'refreshingly': 1.2,
    'pleasantly': 1.2,
    'wonderfully': 1.2,
    'delightfully': 1.2,
    'charmingly': 1.2,
    'appealingly': 1.2,
    'attractively': 1.2,
    'impressively': 1.2,
    'notably': 1.2,
    'conspicuously': 1.2,
    'moderately': 1.1,
    'reasonably': 1.1,
    'relatively': 1.1,
    'comparatively': 1.1,
    'decently': 1.1,
    'acceptably': 1.1,
    'tolerably': 1.1,
    
    # Diminishers (0.3-0.9)
    'somewhat': 0.8,
    'kind of': 0.8,
    'sort of': 0.8,
    'kinda': 0.8,
    'sorta': 0.8,
    'partially': 0.8,
    'half': 0.8,
    'semi': 0.8,
    'quasi': 0.8,
    'pseudo': 0.8,
    'almost': 0.8,
    'nearly': 0.8,
    'near': 0.8,  # Added - similar to "nearly"
    'practically': 0.8,
    'virtually': 0.8,
    'basically': 0.8,
    'essentially': 0.8,
    'mildly': 0.7,
    'slightly': 0.7,
    'minimally': 0.7,
    'marginally': 0.7,
    'lightly': 0.7,
    'gently': 0.7,
    'softly': 0.7,
    'weakly': 0.7,
    'faintly': 0.7,
    'dimly': 0.7,
    'modestly': 0.7,
    'humbly': 0.7,
    'quietly': 0.7,
    'subtly': 0.7,
    'vaguely': 0.6,
    'arguably': 0.6,
    'possibly': 0.6,
    'perhaps': 0.6,
    'maybe': 0.6,
    'potentially': 0.6,
    'conceivably': 0.6,
    'presumably': 0.6,
    'supposedly': 0.6,
    'allegedly': 0.6,
    'reportedly': 0.6,
    'apparently': 0.6,
    'seemingly': 0.6,
    'ostensibly': 0.6,
    'theoretically': 0.6,
    'hypothetically': 0.6,
    'barely': 0.5,
    'just': 0.5,
    'only': 0.5,
    'merely': 0.5,
    'simply': 0.5,
    'purely': 0.5,
    'strictly': 0.5,
    'solely': 0.5,
    'exclusively': 0.5,
    'hardly': 0.4,
    'scarcely': 0.4,
    'rarely': 0.4,
    'seldom': 0.4,
    'insufficiently': 0.4,
    'inadequately': 0.4,
    'nominally': 0.4,
    'superficially': 0.4,
    'artificially': 0.4,
    'mechanically': 0.4,
    'technically': 0.4,
    'formally': 0.4,
    'officially': 0.4,
    'remotely': 0.3,
    'distantly': 0.3,
    'tenuously': 0.3,
    'dubiously': 0.3,
    'questionably': 0.3,
    'uncertainly': 0.3
}

# Define negations - these flip the sentiment
NEGATIONS = {
    # Standard negations
    'not', 'never', 'no', 'none', 'nothing', 'neither', 
    'nobody', 'nowhere', 'cannot', 'without',
    
    # Contractions with 'not'
    "n't", "can't", "won't", "shouldn't", "wouldn't", 
    "couldn't", "doesn't", "don't", "didn't", "isn't", 
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "mustn't", "needn't", "oughtn't", "daren't", "usedn't",
    
    # Negative prefixes and words
    'un', 'dis', 'mis', 'anti', 'non', 'de', 'in', 'im', 'il', 'ir',
    'a', 'an', 'counter', 'contra', 'mal', 'under', 'over',
    'lacking', 'absent', 'missing', 'void', 'devoid', 'empty', 'hollow',
    'deprived', 'bereft', 'barren', 'vacant', 'blank',
    
    # Strong denial and rejection words
    'deny', 'denies', 'denied', 'denying', 'refuse', 'refuses',
    'refused', 'refusing', 'reject', 'rejects', 'rejected', 'rejecting',
    'decline', 'declines', 'declined', 'declining', 'forbid', 'forbids',
    'forbidden', 'forbidding', 'prohibit', 'prohibits', 'prohibited', 'prohibiting',
    'ban', 'banned', 'banning', 'block', 'blocked', 'blocking',
    'prevent', 'prevents', 'prevented', 'preventing', 'stop', 'stops', 'stopped',
    'halt', 'halted', 'cease', 'ceased', 'discontinue', 'discontinued',
    'cancel', 'cancelled', 'canceled', 'canceling', 'cancelling',
    'abandon', 'abandoned', 'abandoning', 'abort', 'aborted', 'aborting',
    
    # Negative expressions and failure words
    'fail', 'fails', 'failed', 'failing', 'failure', 'flop', 'flopped',
    'crash', 'crashed', 'collapse', 'collapsed', 'fall', 'fell', 'fallen',
    'unable', 'impossible', 'unlikely', 'doubtful', 'questionable',
    'hopeless', 'helpless', 'useless', 'worthless', 'pointless', 'meaningless',
    'problem', 'problems', 'trouble', 'troubles', 'difficult', 'difficulty',
    'issue', 'issues', 'complication', 'complications', 'obstacle', 'obstacles',
    'wrong', 'bad', 'poor', 'weak', 'terrible', 'awful', 'horrible',
    'dreadful', 'atrocious', 'appalling', 'shocking', 'disgusting', 'revolting',
    'disappointing', 'disappoints', 'disappointed', 'disappointing',
    'unfortunate', 'unfortunately', 'regret', 'regrets', 'regretful', 'regretfully',
    'sorry', 'apologetic', 'remorseful', 'ashamed', 'embarrassed', 'humiliated',
    
    # Absence and lack words
    'lack', 'lacks', 'lacked', 'lacking', 'shortage', 'shortages',
    'deficit', 'deficits', 'deficient', 'insufficient', 'inadequate',
    'scant', 'scarce', 'sparse', 'limited', 'minimal', 'little',
    'few', 'rare', 'uncommon', 'infrequent', 'occasional', 'sporadic',
    'miss', 'missed', 'missing', 'lose', 'lost', 'losing', 'gone',
    'disappear', 'disappeared', 'vanish', 'vanished', 'extinct',
    
    # Opposition and contrast words
    'against', 'oppose', 'opposes', 'opposed', 'opposing', 'opposition',
    'contrary', 'contrast', 'contrasts', 'contrasting', 'different',
    'unlike', 'opposite', 'reverse', 'inverse', 'counter', 'anti',
    'versus', 'vs', 'challenge', 'challenged', 'challenging', 'confront',
    'conflict', 'conflicts', 'conflicting', 'compete', 'competing', 'rival',
    'resist', 'resisted', 'resisting', 'fight', 'fought', 'fighting',
    'battle', 'struggle', 'struggled', 'struggling', 'combat',
    
    # Dismissive and ignoring words
    'dismiss', 'dismisses', 'dismissed', 'dismissing', 'ignore', 'ignores',
    'ignored', 'ignoring', 'overlook', 'overlooks', 'overlooked', 'overlooking',
    'disregard', 'disregards', 'disregarded', 'disregarding',
    'neglect', 'neglects', 'neglected', 'neglecting', 'skip', 'skipped', 'skipping',
    'avoid', 'avoided', 'avoiding', 'evade', 'evaded', 'evading',
    'escape', 'escaped', 'escaping', 'bypass', 'bypassed', 'bypassing',
    
    # Negative comparatives and superlatives
    'less', 'fewer', 'worse', 'worst', 'inferior', 'below', 'under',
    'beneath', 'lower', 'weaker', 'smaller', 'shorter', 'slower',
    'poorer', 'worse', 'worser', 'lesser', 'minor', 'minimal',
    'reduced', 'decreased', 'diminished', 'lowered', 'downgraded',
    
    # Uncertainty and doubt negations
    'doubt', 'doubts', 'doubted', 'doubting', 'uncertain', 'unsure',
    'unclear', 'ambiguous', 'vague', 'confused', 'confusing',
    'questionable', 'suspicious', 'skeptical', 'dubious', 'hesitant',
    'reluctant', 'unwilling', 'resistant', 'apprehensive', 'worried',
    'concerned', 'anxious', 'nervous', 'troubled', 'disturbed',
    
    # Negative emotions and states
    'hate', 'hated', 'hating', 'hatred', 'despise', 'despised', 'despising',
    'loathe', 'loathed', 'loathing', 'detest', 'detested', 'detesting',
    'dislike', 'disliked', 'disliking', 'resent', 'resented', 'resenting',
    'angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed',
    'frustrated', 'upset', 'disappointed', 'sad', 'depressed', 'miserable',
    'unhappy', 'gloomy', 'pessimistic', 'negative', 'bitter', 'hostile',
    
    # Destructive and harmful words
    'destroy', 'destroyed', 'destroying', 'destruction', 'ruin', 'ruined', 'ruining',
    'damage', 'damaged', 'damaging', 'harm', 'harmed', 'harming', 'harmful',
    'hurt', 'hurting', 'injure', 'injured', 'injuring', 'wound', 'wounded',
    'break', 'broke', 'broken', 'breaking', 'shatter', 'shattered', 'shattering',
    'crush', 'crushed', 'crushing', 'devastate', 'devastated', 'devastating',
    
    # Criticism and condemnation
    'criticize', 'criticized', 'criticizing', 'criticism', 'condemn', 'condemned',
    'blame', 'blamed', 'blaming', 'fault', 'faulted', 'faulting',
    'accuse', 'accused', 'accusing', 'attack', 'attacked', 'attacking',
    'mock', 'mocked', 'mocking', 'ridicule', 'ridiculed', 'ridiculing',
    'insult', 'insulted', 'insulting', 'offend', 'offended', 'offending',
    
    # End and termination words
    'end', 'ended', 'ending', 'finish', 'finished', 'finishing',
    'quit', 'quitted', 'quitting', 'stop', 'stopped', 'stopping',
    'terminate', 'terminated', 'terminating', 'conclude', 'concluded',
}
