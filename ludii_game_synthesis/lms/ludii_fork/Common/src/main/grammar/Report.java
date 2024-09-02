package main.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Report generated by expand/parse/compile process.
 * 
 * @author cambolbro
 */
public class Report
{
	private final List<String> errors   = new ArrayList<String>(); 	
  	private final List<String> warnings = new ArrayList<String>();
  	private final List<String> notes    = new ArrayList<String>();

  	// Ongoing report log to replace printlns to Console
  	private final StringBuilder log = new StringBuilder();
  	
  	private ReportMessenger reportMessageFunctions;
  	
	//-------------------------------------------------------------------------

	public List<String> errors()
	{
		return Collections.unmodifiableList(errors);
	}
	
	public List<String> warnings()
	{
		return Collections.unmodifiableList(warnings);
	}

	public List<String> notes()
	{
		return Collections.unmodifiableList(notes);
	}

	//-------------------------------------------------------------------------
	// Error log
	
	/**
	 * Clear the error log.
	 */
	public void clearLog()
	{
		log.setLength(0);
	}

	/**
	 * Add a string to the error log.
	 */
	public void addLog(final String str)
	{
		log.append(str);
	}

	/**
	 * Add a line to the error log.
	 */
	public void addLogLine(final String str)
	{
		log.append(str + "\n");
	}

	/**
	 * @return Current error log as a string.
	 */
	public String log()
	{
		return log.toString();
	}
	
	//-------------------------------------------------------------------------

	public void addError(final String error)
	{
		if (!errors.contains(error))
			errors.add(error);
	}
	
	public void addWarning(final String warning)
	{
		if (!warnings.contains(warning))
			warnings.add(warning);
	}
	
	public void addNote(final String note)
	{
		if (!notes.contains(note))
			notes.add(note);
	}
		
	//-------------------------------------------------------------------------
	
	public boolean isError()
	{
		return !errors.isEmpty();
	}

	public boolean isWarning()
	{
		return !warnings.isEmpty();
	}
	
	public boolean isNote()
	{
		return !notes.isEmpty();
	}
	
	//-------------------------------------------------------------------------

	public void clear()
	{
		errors.clear();
		warnings.clear();
		notes.clear();
		log.setLength(0);
	}
	
	//-------------------------------------------------------------------------

	/**
	 * @return Input string clipped to maximum char length.
	 */
	public static String clippedString(final String str, final int maxChars)
	{
		if (str.length() < maxChars)
			return str;		
		return str.substring(0, maxChars-3) + "...";
	}
	
	//-------------------------------------------------------------------------

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder();

		for (final String error : errors)
			sb.append(error);
		
		for (final String warning : warnings)
			sb.append("Warning: " + warning);
		
		for (final String note : notes)
			sb.append("Note: " + note);
		
		return sb.toString();
				
	}
	
	//-------------------------------------------------------------------------
	
	/**
	 * Report Messenger interface for printing report messages as needed.
	 * 
	 * @author Matthew.Stephenson
	 */
	public interface ReportMessenger 
	{
		void printMessageInStatusPanel(String s);
		void printMessageInAnalysisPanel(String s);
	}

	public ReportMessenger getReportMessageFunctions() 
	{
		return reportMessageFunctions;
	}

	public void setReportMessageFunctions(final ReportMessenger reportMessageFunctions) 
	{
		this.reportMessageFunctions = reportMessageFunctions;
	}
	
	//-------------------------------------------------------------------------

}
