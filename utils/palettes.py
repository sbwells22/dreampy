from matplotlib.colors import LinearSegmentedColormap

spalette = [
    "#F1C40F", 
    "#003366", 
    "#C0392B", 
    "#16A085", 
    "#D35400", 
    "#AAB7B8", 
    "#5B2A6F", 
    "#F39C12", 
    "#1F618D", 
    "#8E4E3A", 
    "#2980B9", 
    "#7B241C", 
    "#A569BD", 
    "#BDC3C7", 
    "#5D4037", 
    "#E74C3C", 
    "#566573", 
    "#D68910", 
    "#922B21", 
    "#117864", 
    "#AF7AC5", 
    "#2C3E50", 
    "#ECF0F1", 
    "#7F8C8D", 
    "#5B2C6F", 
    "#1ABC9C", 
    "#3498DB", 
    "#34495E"
]

spalette1 = [
    "#C0392B",  # Deep Red
    "#F39C12",  # Mustard Yellow
    "#2980B9",  # Muted Blue
    "#16A085",  # Teal Green
    "#7B241C",  # Dark Brown
    "#ECF0F1",  # Off-White
    "#2C3E50",  # Charcoal Black
    "#AAB7B8",  # Medium Gray 
    "#A569BD",  # Muted Purple
    "#5B2A6F" , # Darker Muted Purple
    "#003366",  # Muted Deep Navy Blue
]

spalette2 = [
    "#C0392B",  # Deep Red
    "#E74C3C",  # Light Red
    "#922B21",  # Darker Red
    "#F39C12",  # Mustard Yellow
    "#F1C40F",  # Bright Yellow
    "#D68910",  # Darker Yellow
    "#2980B9",  # Muted Blue
    "#3498DB",  # Light Blue
    "#1F618D",  # Dark Blue
    "#16A085",  # Teal Green
    "#1ABC9C",  # Light Teal
    "#117864",  # Dark Teal
    "#7B241C",  # Dark Brown
    "#8E4E3A",  # Reddish Brown
    "#5D4037",  # Cooler Brown
    "#ECF0F1",  # Off-White
    "#BDC3C7",  # Light Gray
    "#AAB7B8",  # Medium Gray
    "#2C3E50",  # Charcoal Black
    "#34495E",  # Dark Gray
    "#566573",  # Steel Gray
    "#7F8C8D",  # Blue Gray
    "#AF7AC5",  # Purple
    "#5B2C6F",  # Dark Purple
    "#D35400",  # Burnt Orange
]

scmap_colors = ['#2C3E50',"#003366",'#2980B9','#3498DB','#ECF0F1','#F1C40F','#F39C12','#E74C3C','#4A0E12']
scmap = LinearSegmentedColormap.from_list("scmap", scmap_colors)
scmap.set_bad(color='#34495E', alpha=0.05)

bwr_colors =['#003366',"#3498DB",'#ECF0F1','#E74C3C','#7B241C']
bwr_cmap = LinearSegmentedColormap.from_list("bwr_cmap", bwr_colors)
# bwr_cmap.set_bad(color='#34495E', alpha=0.05)

bwbr_colors =['#003366',"#3498DB",'#ECF0F1','#F39C12','#7B241C']
bwbr_cmap = LinearSegmentedColormap.from_list("bwbr_cmap", bwbr_colors)
bwbr_cmap.set_bad(color='#34495E', alpha=0.05)

warm_colors =['#ECF0F1', '#ECF0F1', '#E74C3C', '#C0392B', '#7B241C', '#4A0E12']
warm = LinearSegmentedColormap.from_list("warm", warm_colors)
warm.set_bad(color='#34495E', alpha=0.05)

cool_colors = ['#ECF0F1', '#ECF0F1', '#3498DB', '#2980B9', '#003366', '#2C3E50']
cool = LinearSegmentedColormap.from_list("cool", cool_colors)
cool.set_bad(color='#34495E', alpha=0.05)

scmap_warm_colors = ['#ECF0F1','#ECF0F1','#F1C40F','#F39C12','#E74C3C','#4A0E12']
scmap_warm = LinearSegmentedColormap.from_list("scmap_warm", scmap_warm_colors)
scmap_warm.set_bad(color='#34495E', alpha=0.05)
