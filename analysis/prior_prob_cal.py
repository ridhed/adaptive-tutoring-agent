import pandas as pd

def calculate_global_bkt_params(csv_path="skill_builder_data.csv"):

    columns_needed = ['user_id', 'skill_name', 'correct', 'opportunity', 'order_id']
    df = pd.read_csv(csv_path, usecols=columns_needed)
    df = df.dropna(subset=['user_id', 'skill_name', 'correct', 'opportunity'])
    df = df.sort_values(['user_id', 'skill_name', 'order_id'])
    
    p_initial = df[df['opportunity'] == 1]['correct'].mean()
    
    df['prev_correct'] = df.groupby(['user_id', 'skill_name'])['correct'].shift(1)
    df['prev2_correct'] = df.groupby(['user_id', 'skill_name'])['correct'].shift(2)
    
    guess_mask = (df['prev_correct'] == 0) & (df['prev2_correct'] == 0)
    p_guess = df[guess_mask]['correct'].mean()
    
    slip_mask = (df['prev_correct'] == 1) & (df['prev2_correct'] == 1)
    p_slip = 1.0 - df[slip_mask]['correct'].mean()
    
    raw_transition = df[df['prev_correct'] == 0]['correct'].mean()
    p_learn = max(0.01, raw_transition - p_guess) 
    
    return {
        "p_initial_know": round(p_initial, 3),
        "p_guess": round(p_guess, 3),
        "p_slip": round(p_slip, 3),
        "p_learn": round(p_learn, 3),
        "p_forget": 0.0
    }

if __name__ == "__main__":
    learned_params = calculate_global_bkt_params("skill_builder_data.csv")
    print("Calculated Parameters from Dataset:")
    print(learned_params)