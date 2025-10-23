
class DatasetManager:
    """Manage and combine multiple dataset sources."""
    
    def __init__(self):
        self.generators = []
        
    def add_generator(self, generator: DatasetGenerator, weight: float = 1.0):
        """Add a dataset generator with specified weight."""
        self.generators.append((generator, weight))
        
    def generate_balanced_dataset(
        self, 
        total_transitions: int,
        min_unsafe_ratio: float = 0.1,
        min_goal_ratio: float = 0.05
    ) -> List[Transition]:
        """Generate balanced dataset with specified ratios."""
        
        # Calculate transitions per generator
        total_weight = sum(weight for _, weight in self.generators)
        transitions_per_gen = [
            int(total_transitions * weight / total_weight) 
            for _, weight in self.generators
        ]
        
        # Generate transitions from each generator
        all_transitions = []
        for (generator, _), num_trans in zip(self.generators, transitions_per_gen):
            transitions = generator.generate_transitions(num_trans)
            all_transitions.extend(transitions)
            
        # Analyze current distribution
        safe_count = sum(1 for t in all_transitions if t.is_safe)
        unsafe_count = len(all_transitions) - safe_count
        goal_count = sum(1 for t in all_transitions if t.is_goal)
        
        # Add more unsafe transitions if needed
        target_unsafe = max(int(total_transitions * min_unsafe_ratio), unsafe_count)
        if unsafe_count < target_unsafe:
            needed_unsafe = target_unsafe - unsafe_count
            # Generate more transitions and filter for unsafe ones
            # (Implementation would depend on specific generators)
            
        # Add more goal transitions if needed
        target_goal = max(int(total_transitions * min_goal_ratio), goal_count)
        if goal_count < target_goal:
            needed_goal = target_goal - goal_count
            # Generate more transitions and filter for goal ones
            # (Implementation would depend on specific generators)
            
        # Shuffle and return
        random.shuffle(all_transitions)
        return all_transitions[:total_transitions]