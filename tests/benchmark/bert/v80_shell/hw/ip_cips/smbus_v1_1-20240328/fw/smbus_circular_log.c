/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains functions to initialize and to add log entries to the SMBus Driver circular log
 * for the SMBus driver.
 *
 * @file smbus_circular_log.c
 *
 */

#include "smbus_internal.h"
#include "smbus.h"
#include "smbus_state.h"
#include "smbus_event.h"

#define SMBUS_LOG_IS_OCCUPIED               ( 0xAAABACAD )
#define SMBUS_LOG_IS_NOT_OCCUPIED           ( 0 )

/********************** Static function declarations ***************************/

/******************************************************************************
*
* @brief    Is a conversion function from the state machine event to character string
*           to be used by logging functions
*
* @param    xEvent is any event handled by the state machine
*
* @return   A character string
*
* @note     None.
*
*****************************************************************************/
static char* prvpcConvertEventTypeToText( SMBUS_LOG_EVENT_TYPE xEvent );

/******************************************************************************
*
* @brief    This function formats a log entry as a text string ready to be displayed
*           The format of the string depends on the type of event that was logged
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure
* @param    entry is the index of the log entry in the buffer
* @param    pcLogBuffer is a char buffer containing the complete log to display
* @param    pslLineSize is pointer to the size of the log string being added
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvFormatLine( SMBUS_PROFILE_TYPE* pxSMBusProfile, int entry, char* pcLogBuffer,
                                int* pslLineSize );

/*******************************************************************************/

/******************************************************************************
*
* @brief    This function formats a log entry as a text string ready to be displayed
*           The format of the string depends on the type of event that was logged
*
*****************************************************************************/
static char* prvpcConvertEventTypeToText( SMBUS_LOG_EVENT_TYPE xEvent )
{
    char* pcReturnText = "         ";   
    
    if( SMBUS_LOG_EVENT_INTERRUPT_EVENT == xEvent )
    {
        pcReturnText = "INTERRUPT";
    }
    else if( SMBUS_LOG_EVENT_FSM_EVENT == xEvent )
    {
        pcReturnText = "FSM      ";
    }
    else if( SMBUS_LOG_EVENT_ERROR == xEvent )
    {
        pcReturnText = "ERROR    ";
    }
    else if( SMBUS_LOG_EVENT_HW_READ == xEvent )
    {
        pcReturnText = "HW_READ  ";
    }
    else if( SMBUS_LOG_EVENT_HW_WRITE == xEvent )
    {
        pcReturnText = "HW_WRITE ";
    }
    else if( SMBUS_LOG_EVENT_PROTOCOL == xEvent )
    {
        pcReturnText = "PROTOCOL ";
    }
    else if( SMBUS_LOG_EVENT_DEBUG == xEvent )
    {
        pcReturnText = "DEBUG    ";
    }
    else if( SMBUS_LOG_EVENT_TRYREAD == xEvent )
    {
        pcReturnText = "TRYREAD  ";
    }
    else if( SMBUS_LOG_EVENT_TRYWRITE == xEvent )
    {
        pcReturnText = "TRYWRITE ";
    }
    
    return (pcReturnText);
}

/******************************************************************************
*
* @brief    This function formats a log entry as a text string ready to be displayed
*           The format of the string depends on the type of event that was logged
*
*****************************************************************************/
static void prvvFormatLine( SMBUS_PROFILE_TYPE* pxSMBusProfile, int entry, char* pcLogBuffer, 
                                int* pslLineSize )
{
    char* pcState       = NULL;
    char* pcEvent       = NULL;
    char* pcProtocol    = NULL;

    if( ( NULL != pxSMBusProfile ) && 
        ( NULL != pcLogBuffer )    &&
        ( NULL != pslLineSize ) )
    {
        switch( pxSMBusProfile->xCircularBuffer[entry].xEvent )
        {
        case SMBUS_LOG_EVENT_TRYWRITE:          /* Fall through deliberate */
        case SMBUS_LOG_EVENT_TRYREAD:           /* Fall through deliberate */
        case SMBUS_LOG_EVENT_DEBUG:
            *pslLineSize = sprintf( pcLogBuffer, "%04d %07d %s %2d 0x%08x line %d\r\n", entry,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulTicks,
            prvpcConvertEventTypeToText( pxSMBusProfile->xCircularBuffer[entry].xEvent ),
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulInstance,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulEntry1,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulEntry2 );
            break;

        case SMBUS_LOG_EVENT_PROTOCOL:
            pcProtocol = pcProtocolToString( pxSMBusProfile->xCircularBuffer[entry].ulEntry2 );
            *pslLineSize = sprintf( pcLogBuffer, "%04d %07d %s %2d 0x%08x %s\r\n", entry,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulTicks,
            prvpcConvertEventTypeToText( pxSMBusProfile->xCircularBuffer[entry].xEvent ),
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulInstance,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulEntry1,
            pcProtocol );
            break;

        case SMBUS_LOG_EVENT_HW_WRITE:          /* Fall through deliberate */
        case SMBUS_LOG_EVENT_HW_READ:           /* Fall through deliberate */
        case SMBUS_LOG_EVENT_INTERRUPT_EVENT:
            *pslLineSize = sprintf( pcLogBuffer, "%04d %07d %s %2d 0x%08x 0x%08x\r\n", entry,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulTicks,
            prvpcConvertEventTypeToText( pxSMBusProfile->xCircularBuffer[entry].xEvent ),
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulInstance,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulEntry1,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulEntry2 );
            break;

        case SMBUS_LOG_EVENT_ERROR:         /* Fall through deliberate */
        case SMBUS_LOG_EVENT_FSM_EVENT:
            pcState = ( char* )pcStateToString( pxSMBusProfile->xCircularBuffer[entry].ulEntry1 );
            pcEvent = ( char* )pcEventToString( pxSMBusProfile->xCircularBuffer[entry].ulEntry2 );
            *pslLineSize = sprintf( pcLogBuffer, "%04d %07d %s %2d %s %s\r\n", entry,
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulTicks,
            prvpcConvertEventTypeToText( pxSMBusProfile->xCircularBuffer[entry].xEvent ),
            ( unsigned int )pxSMBusProfile->xCircularBuffer[entry].ulInstance, pcState, pcEvent );
            break;
                    
        default:
            *pslLineSize = 0;
            break;          
        }
    }
}

/*******************************************************************************
*
* @brief    Will retreive the log as a character string
*
*******************************************************************************/
void vLogDisplayLog( SMBUS_PROFILE_TYPE* pxSMBusProfile, char* pcLogBuffer, uint32_t* usLogSizeBytes )
{
    int slStart         = 0;
    int i               = 0;
    int slLineSize      = 0;
    uint32_t usLogSize  = 0;

    if( ( NULL != pxSMBusProfile ) && 
        ( NULL != pcLogBuffer )    &&
        ( NULL != usLogSizeBytes ) )
    {
        slStart = pxSMBusProfile->xLogCircularBuffer.ulWrite;

        for( i = slStart; i < SMBUS_MAX_CIRCULAR_LOG_ENTRIES; i++ )
        {
            if( SMBUS_LOG_IS_OCCUPIED == pxSMBusProfile->xCircularBuffer[i].ulIsOccupied )
            {
                prvvFormatLine( pxSMBusProfile, i, (pcLogBuffer + usLogSize), &slLineSize );
                usLogSize += ( uint32_t )slLineSize;
            }
        }

        for( i = 0; i < slStart; i++ )
        {
            if( SMBUS_LOG_IS_OCCUPIED == pxSMBusProfile->xCircularBuffer[i].ulIsOccupied )
            {
                prvvFormatLine( pxSMBusProfile, i, ( pcLogBuffer + usLogSize ), &slLineSize );
                usLogSize += ( uint32_t )slLineSize;
            }
        }
        
        *usLogSizeBytes = usLogSize;
    }
}

/*******************************************************************************
*
* @brief    Initializes the debug log. Setting its pointer to zero
*
*******************************************************************************/
void vLogInitialize( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t i = 0;

    if( NULL != pxSMBusProfile )    
    {
        for( i = 0; i < ( SMBUS_MAX_CIRCULAR_LOG_ENTRIES ); i++ )
        {
            pxSMBusProfile->xCircularBuffer[i].ulIsOccupied = SMBUS_LOG_IS_NOT_OCCUPIED;
            pxSMBusProfile->xCircularBuffer[i].ulEntry1 = 0x00;
            pxSMBusProfile->xCircularBuffer[i].ulEntry2 = 0x00;
            pxSMBusProfile->xCircularBuffer[i].ulTicks = 0x00;
        }
        
        pxSMBusProfile->xLogCircularBuffer.ulWrite = 0;
        pxSMBusProfile->xLogCircularBuffer.ulRead = 0;
    }
}

/*******************************************************************************
*
* @brief    Will add a log entry into the debug log
*
*******************************************************************************/
void vLogAddEntry( SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBUS_LOG_LEVEL_TYPE xLogLevel, uint32_t ulInstance,
                    SMBUS_LOG_EVENT_TYPE  Log_Event, uint32_t ulEntry1, uint32_t ulEntry2 )
{
    uint32_t ulTicks  = 0;

    if( NULL != pxSMBusProfile )  
    {
        if( xLogLevel <= pxSMBusProfile->xLogLevel )
        {
            if( NULL != pxSMBusProfile->pFnReadTicks )
            {
                pxSMBusProfile->pFnReadTicks( &ulTicks );
            }

            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].ulTicks = ulTicks;
            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].xEvent = Log_Event;
            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].ulInstance = ulInstance;
            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].ulEntry1 = ulEntry1;
            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].ulEntry2 = ulEntry2;
            pxSMBusProfile->xCircularBuffer[pxSMBusProfile->xLogCircularBuffer.ulWrite].ulIsOccupied = SMBUS_LOG_IS_OCCUPIED;

            if( (SMBUS_MAX_CIRCULAR_LOG_ENTRIES - 1)  <= pxSMBusProfile->xLogCircularBuffer.ulWrite )
            {
                /* For debug Just write over last log */
                pxSMBusProfile->xLogCircularBuffer.ulWrite = ( SMBUS_MAX_CIRCULAR_LOG_ENTRIES - 1 );
                /* pxSMBusProfile->xLogCircularBuffer.ulWrite = 0; */
            }
            else
            {
                pxSMBusProfile->xLogCircularBuffer.ulWrite++;
            }
        }
    }
}
