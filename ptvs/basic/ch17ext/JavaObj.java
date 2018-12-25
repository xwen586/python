public class JavaObj{
    String value;
	
    public JavaObj(String value)
    { 
        this.value = value + " Java"; 
    } 
    public String getValue()
    { 
        return this.value;
    }
    public void say()
    {
        System.out.println("hello java");
    }
	
    public static void main(String[] args) {
        System.out.println("Hello java world");    
    }
 }
 